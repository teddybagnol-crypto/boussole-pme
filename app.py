import os
import io
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import groq
from datetime import datetime
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify, redirect, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

load_dotenv()
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'boussole-secret-dev-key')

_db_url = os.environ.get('DATABASE_URL', 'sqlite:///boussole.db')
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
app.config['SESSION_PERMANENT'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

app.config['SESSION_TYPE'] = 'sqlalchemy'
app.config['SESSION_SQLALCHEMY'] = db
Session(app)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')


# ─────────────────────────────────────────
# MODÈLES BASE DE DONNÉES
# ─────────────────────────────────────────
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    entreprise = db.Column(db.String(150))

class KpiSuivi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    nom = db.Column(db.String(150), nullable=False)
    ordre = db.Column(db.Integer, default=0)

class PeriodeEntreprise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date_upload = db.Column(db.DateTime, default=datetime.utcnow)
    label_periode = db.Column(db.String(100))
    score = db.Column(db.Float)
    note = db.Column(db.String(1))
    resume_ia = db.Column(db.Text)

class ValeurKpi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    periode_id = db.Column(db.Integer, db.ForeignKey('periode_entreprise.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    nom_kpi = db.Column(db.String(150), nullable=False)
    valeur = db.Column(db.Float)


# ─────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def lire_fichier(file):
    filename = file.filename.lower()
    file_bytes = file.read()
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        return pd.read_excel(io.BytesIO(file_bytes))
    elif filename.endswith('.csv'):
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            for sep in [',', ';']:
                try:
                    tmp = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=sep)
                    if tmp.shape[1] > 1:
                        df = tmp
                        break
                except Exception:
                    continue
            if df is not None:
                break
        if df is None:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python')
        return df
    else:
        raise ValueError('Format non supporté.')

def detecter_date(df):
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().sum() > len(df) * 0.5:
                return c
        except Exception:
            continue
    return None

def calculer_score(kpis_valeurs, kpis_precedents=None):
    if not kpis_valeurs:
        return 50.0, 'C'
    score = 50.0
    if kpis_precedents:
        progressions = []
        for nom, val in kpis_valeurs.items():
            if nom in kpis_precedents and kpis_precedents[nom] and kpis_precedents[nom] != 0:
                delta_pct = (val - kpis_precedents[nom]) / abs(kpis_precedents[nom]) * 100
                progressions.append(delta_pct)
        if progressions:
            moy_progression = np.mean(progressions)
            bonus = min(40, max(-40, moy_progression * 2))
            score = 50 + bonus
            if all(p >= 0 for p in progressions):
                score += 10
    else:
        score = 60.0
    score = max(0, min(100, score))
    if score >= 80:
        note = 'A'
    elif score >= 65:
        note = 'B'
    elif score >= 45:
        note = 'C'
    elif score >= 25:
        note = 'D'
    else:
        note = 'E'
    return round(score, 1), note


# Mots-clés qui identifient une variable temporelle/de date non actionnable
_DATE_LIKE = ['année', 'annee', 'year', 'date', 'mois', 'month', 'jour', 'day',
              'trimestre', 'quarter', 'semaine', 'week', 'période', 'periode',
              'heure', 'hour', 'minute', 'time', 'timestamp']

def est_date_like(nom):
    n = nom.lower().strip()
    return any(p in n for p in _DATE_LIKE)


def calculer_importances_kpis(user_id):
    """Priorité 1 : RF (≥5 périodes) ou corrélation Pearson (3-4 périodes) sur l'historique."""
    periodes = PeriodeEntreprise.query.filter_by(user_id=user_id).order_by(PeriodeEntreprise.date_upload).all()
    if len(periodes) < 3:
        return {}
    rows = []
    for p in periodes:
        kpis = {v.nom_kpi: v.valeur for v in ValeurKpi.query.filter_by(periode_id=p.id, user_id=user_id).all()}
        kpis_clean = {k: v for k, v in kpis.items() if not est_date_like(k) and v is not None}
        rows.append({'_score': p.score, **kpis_clean})
    if not rows:
        return {}
    df = pd.DataFrame(rows).dropna()
    kpi_cols = [c for c in df.columns if c != '_score' and df[c].std() > 0]
    if len(df) < 3 or not kpi_cols:
        return {}
    try:
        if len(df) >= 5:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(df[kpi_cols].values, df['_score'].values)
            raw = dict(zip(kpi_cols, rf.feature_importances_))
        else:
            raw = {}
            for col in kpi_cols:
                c = float(df[col].corr(df['_score']))
                raw[col] = abs(c) if not np.isnan(c) else 0.0
        total = sum(raw.values()) or 1
        return {k: round(v / total, 4) for k, v in raw.items()}
    except Exception:
        return {}


def importances_depuis_session(kpis_cibles):
    """Priorité 2 : utilise les importances RF calculées par l'Analyseur (session Flask).
    Fait correspondre les noms de colonnes de l'analyse aux KPIs de l'évolution.
    Retourne {} si aucune correspondance suffisante (< 50% de recouvrement)."""
    import re
    rf_summary = session.get('rf_summary', '')
    top3 = session.get('top3_factors', [])
    if not rf_summary and not top3:
        return {}

    # Parse "col:XX.X%." depuis rf_summary (format généré par /analyse)
    raw = {}
    for nom, pct in re.findall(r'([^:\s][^:]*):(\d+\.?\d*)%', rf_summary):
        raw[nom.strip()] = float(pct)

    # Si le parsing échoue, utiliser top3 avec poids décroissants 4-2-1
    if not raw and top3:
        poids = [4, 2, 1]
        for i, col in enumerate(top3[:3]):
            raw[col] = float(poids[i])

    if not raw:
        return {}

    # Correspondance exacte puis insensible à la casse
    matched = {}
    kpis_lower = {k.lower(): k for k in kpis_cibles}
    for col, imp in raw.items():
        if col in kpis_cibles:
            matched[col] = imp
        elif col.lower() in kpis_lower:
            matched[kpis_lower[col.lower()]] = imp

    # Recouvrement insuffisant : on refuse (évite d'appliquer des poids hors-sujet)
    if len(matched) < max(1, len(kpis_cibles) * 0.5):
        return {}

    # Les KPIs sans correspondance reçoivent la moyenne des importances trouvées
    moy = sum(matched.values()) / len(matched)
    for k in kpis_cibles:
        if k not in matched:
            matched[k] = moy * 0.3  # poids réduit car non observé dans l'analyse

    total = sum(matched.values()) or 1
    return {k: round(v / total, 4) for k, v in matched.items()}


def resoudre_importances(user_id, kpis_cibles):
    """Cascade de résolution des importances — retourne toujours un dict normalisé.
    Source 1 : historique périodes (RF / Pearson).
    Source 2 : session Analyseur (RF sur dataset complet).
    Source 3 : poids égaux (dernier recours explicitement signalé)."""
    imp = calculer_importances_kpis(user_id)
    source = 'historique'
    if not imp or not any(k in imp for k in kpis_cibles):
        imp = importances_depuis_session(kpis_cibles)
        source = 'analyseur'
    if not imp:
        n = len(kpis_cibles) or 1
        imp = {k: round(1/n, 4) for k in kpis_cibles}
        source = 'egal'

    # Restreindre et renormaliser sur les KPIs cibles
    imp = {k: imp.get(k, 0.0) for k in kpis_cibles}
    total = sum(imp.values()) or 1
    imp = {k: round(v / total, 4) for k, v in imp.items()}
    return imp, source


# ─────────────────────────────────────────
# ROUTES AUTH
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inscription', methods=['GET', 'POST'])
def inscription():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        entreprise = request.form.get('entreprise')
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email déjà utilisé'})
        user = User(
            email=email,
            password=generate_password_hash(password),
            entreprise=entreprise
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return jsonify({'success': True, 'redirect': '/dashboard'})
    return render_template('inscription.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return jsonify({'success': True, 'redirect': '/dashboard'})
        return jsonify({'success': False, 'error': 'Email ou mot de passe incorrect'})
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect('/')

@app.route('/dashboard')
@login_required
def dashboard():
    session['chat_history'] = []
    return render_template('dashboard.html', user=current_user)


# ─────────────────────────────────────────
# ROUTE ANALYSEUR
# ─────────────────────────────────────────
@app.route('/analyse', methods=['POST'])
@login_required
def analyse():
    try:
        if 'chat_history' not in session:
            session['chat_history'] = []

        question = request.form.get('question', '')
        file_present = 'file' in request.files and request.files['file'].filename != ''

        if file_present:
            session['chat_history'] = []
            file = request.files['file']
            df = lire_fichier(file)

            if df is None or df.empty:
                return jsonify({'success': False, 'error': 'Fichier vide ou illisible.'})

            session['all_columns'] = list(df.columns)[:10]
            df_numeric = df.select_dtypes(include=['number'])
            numeric_columns = list(df_numeric.columns)
            session['len_df'] = len(df)
            session['stats'] = df.describe().round(1).to_string()[:300]
            session['correlation_matrix'] = df_numeric.corr().round(2).to_string()[:200]

            if len(numeric_columns) < 2:
                return jsonify({'success': False, 'error': "Minimum 2 colonnes numériques requises."})

            client = groq.Groq(api_key=GROQ_API_KEY)
            choix_cible = client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                response_format={"type": "json_object"},
                messages=[{'role': 'user', 'content': f'''Identifie l unique variable dependante Y.
Reponds en json : {{"Y": "nom_colonne"}}
Question : "{question}"
Colonnes : {numeric_columns}'''}]
            )
            choix = json.loads(choix_cible.choices[0].message.content)
            var_Y = choix.get('Y')
            if var_Y not in df_numeric.columns:
                var_Y = numeric_columns[0]
            session['var_Y'] = var_Y

            X_columns = [col for col in numeric_columns if col != var_Y]
            Y = df_numeric[var_Y].values
            X_df = df_numeric[X_columns].fillna(0)

            # MCO
            try:
                X_ols = sm.add_constant(X_df)
                model_ols = sm.OLS(Y, X_ols).fit()
                session['r_sq'] = round(float(model_ols.rsquared), 4)
                eco = f"MCO R²={model_ols.rsquared:.3f}. "
                for col in X_columns[:3]:
                    sig = "sig" if model_ols.pvalues[col] < 0.05 else "non-sig"
                    eco += f"{col}:coef={model_ols.params[col]:.2f},p={model_ols.pvalues[col]:.3f}({sig}). "
                session['regression_summary'] = eco[:300]
            except Exception as e:
                session['r_sq'] = 0.0
                session['regression_summary'] = f"Erreur MCO:{str(e)[:50]}"

            # Random Forest
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_df)
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_scaled, Y)
                importances = dict(zip(X_columns, rf.feature_importances_))
                importances_sorted = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                cv_scores = cross_val_score(rf, X_scaled, Y, cv=min(5, len(Y)//2), scoring='r2')
                rf_r2 = round(float(np.mean(cv_scores)), 4)
                rf_txt = f"RF R²={rf_r2:.3f}. "
                for col, imp in importances_sorted[:3]:
                    rf_txt += f"{col}:{imp*100:.1f}%. "
                session['rf_summary'] = rf_txt[:300]
                session['rf_r2'] = rf_r2
                session['top3_factors'] = [col for col, _ in importances_sorted[:3]]
            except Exception as e:
                session['rf_summary'] = f"Erreur RF:{str(e)[:50]}"
                session['rf_r2'] = 0.0
                session['top3_factors'] = X_columns[:3]

            # LASSO
            try:
                lasso = LassoCV(cv=min(5, len(Y)//2), random_state=42, max_iter=10000)
                lasso.fit(X_scaled, Y)
                elim = [col for col, coef in zip(X_columns, lasso.coef_) if abs(coef) <= 0.001]
                keep = [col for col, coef in zip(X_columns, lasso.coef_) if abs(coef) > 0.001]
                session['lasso_summary'] = f"LASSO garde:{keep[:3]}, elimine:{elim[:3]}"[:200]
            except Exception as e:
                session['lasso_summary'] = f"Erreur LASSO:{str(e)[:50]}"

            # Causal Forest
            try:
                treatment_var = session['top3_factors'][0] if session.get('top3_factors') else X_columns[0]
                T = X_df[treatment_var].values
                T_binary = (T > np.median(T)).astype(int)
                ate = float(np.mean(Y[T_binary==1]) - np.mean(Y[T_binary==0]))
                cf_txt = f"CF traitement:{treatment_var}, ATE:{ate:.2f}. "
                if len(X_columns) > 1:
                    mod = X_df[X_columns[1]].values
                    high = mod > np.median(mod)
                    ate_h = float(np.mean(Y[T_binary==1][high[T_binary==1]]) - np.mean(Y[T_binary==0][high[T_binary==0]])) if sum(high) > 0 else 0
                    cf_txt += f"Effet haut:{ate_h:.2f}."
                session['cf_summary'] = cf_txt[:200]
                session['ate'] = f"{ate:.2f}"
                session['treatment_var'] = treatment_var
            except Exception as e:
                session['cf_summary'] = f"Erreur CF:{str(e)[:50]}"
                session['ate'] = "N/A"
                session['treatment_var'] = "N/A"

            # Anomalies
            try:
                anomalies = {
                    col: int(len(df_numeric[
                        (df_numeric[col] > df_numeric[col].mean() + 2*df_numeric[col].std()) |
                        (df_numeric[col] < df_numeric[col].mean() - 2*df_numeric[col].std())
                    ]))
                    for col in numeric_columns[:5]
                }
                anomalies = {k: v for k, v in anomalies.items() if v > 0}
                session['anomalies'] = str(anomalies)[:150]
            except Exception:
                session['anomalies'] = "{}"

            # Projection
            try:
                derniere = df_numeric[var_Y].iloc[-1]
                croissance = df_numeric[var_Y].pct_change().mean()
                session['projection_3mois'] = f"{derniere * (1 + croissance)**3:.0f}"
                session['croissance_moy'] = f"{croissance*100:.1f}%"
            except Exception:
                session['projection_3mois'] = "N/A"
                session['croissance_moy'] = "0%"

        else:
            if 'stats' not in session:
                return jsonify({'success': False, 'error': "Veuillez d'abord charger un fichier."})

        client = groq.Groq(api_key=GROQ_API_KEY)
        history = session.get('chat_history', [])

        system_instruction = f'''Tu es Data Scientist et Consultant chez Boussole. Reponds en json valide uniquement.
DONNEES : colonnes={session.get('all_columns')}, lignes={session.get('len_df')}, Y={session.get('var_Y')}
MCO : {session.get('regression_summary')}
RF : {session.get('rf_summary')}
LASSO : {session.get('lasso_summary')}
CF : {session.get('cf_summary')} ATE={session.get('ate')}
Anomalies : {session.get('anomalies')}
Projection 3 mois : {session.get('projection_3mois')}
Question posee par le dirigeant : "{question}"
CONSIGNES : json valide. Croise MCO+RF+CF. Actions concretes chiffrees. Effort Faible/Moyen/Fort.
FORMAT json :
{{
  "metrics":[
    {{"label":"R2 MCO","value":"{session.get('r_sq',0):.2f}","sub":"Econometrie"}},
    {{"label":"R2 Random Forest","value":"{session.get('rf_r2',0):.2f}","sub":"Machine Learning"}},
    {{"label":"Effet Causal ATE","value":"{session.get('ate')}","sub":"Causal Forest"}}
  ],
  "diagnostic":{{"tendances":[{{"facteur":"...","impact":"+X%","description":"..."}}],"anomalie":"...","correlation":"..."}},
  "predictif":{{"projection":"...","probabilite_objectif":"XX%","scenario_optimiste":"...","scenario_pessimiste":"..."}},
  "recommandations":[{{"priorite":"1","action":"...","detail":"... Effort Faible"}}],
  "kpis":[{{"nom":"...","valeur_actuelle":"...","valeur_cible":"...","statut":"normal"}}],
  "analyse":"3 phrases pour le decideur.",
  "synthese_finale":"Paragraphe detaille de 8 a 10 phrases qui repond directement et precisement a la question posee par le dirigeant. Commence par rappeler la question. Cite les chiffres cles : R2 MCO={session.get('r_sq',0):.2f}, R2 RF={session.get('rf_r2',0):.2f}, ATE={session.get('ate')}. Explique ce que chaque modele apporte comme eclairage. Mentionne les variables les plus significatives et leur impact chiffre. Conclut avec une recommandation concrete et chiffree directement liee a la question."
}}'''

        messages_to_send = [{'role': 'system', 'content': system_instruction}]
        for msg in history:
            messages_to_send.append(msg)
        messages_to_send.append({'role': 'user', 'content': question})

        completion = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            response_format={"type": "json_object"},
            messages=messages_to_send
        )
        raw_response = completion.choices[0].message.content
        result = json.loads(raw_response)
        history.append({'role': 'user', 'content': question})
        history.append({'role': 'assistant', 'content': raw_response[:300]})
        session['chat_history'] = history[-4:]
        session.modified = True
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        print("ERREUR /analyse :", str(e))
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# ÉVOLUTION — RÉCUPÉRER LES DONNÉES
# ─────────────────────────────────────────
@app.route('/evolution_data', methods=['GET'])
@login_required
def evolution_data():
    try:
        kpis_suivi = KpiSuivi.query.filter_by(user_id=current_user.id).order_by(KpiSuivi.ordre).all()
        kpis_noms = [k.nom for k in kpis_suivi]
        periodes = PeriodeEntreprise.query.filter_by(user_id=current_user.id).order_by(PeriodeEntreprise.date_upload).all()
        historique = []
        for p in periodes:
            valeurs = ValeurKpi.query.filter_by(periode_id=p.id, user_id=current_user.id).all()
            vals_dict = {v.nom_kpi: v.valeur for v in valeurs}
            historique.append({
                'id': p.id,
                'date': p.date_upload.strftime('%d/%m/%Y'),
                'label': p.label_periode,
                'score': p.score,
                'note': p.note,
                'resume_ia': p.resume_ia,
                'valeurs': vals_dict
            })
        # Averages
        averages = {}
        for kpi in kpis_noms:
            vals = [p['valeurs'][kpi] for p in historique if kpi in p['valeurs'] and p['valeurs'][kpi] is not None]
            if vals:
                averages[kpi] = round(sum(vals) / len(vals), 2)

        # Linear regression predictions M+1, M+2, M+3
        predictions = {}
        if len(historique) >= 2:
            for kpi in kpis_noms:
                pts = [(i, p['valeurs'][kpi]) for i, p in enumerate(historique)
                       if kpi in p['valeurs'] and p['valeurs'][kpi] is not None]
                if len(pts) >= 2:
                    x_arr = np.array([t[0] for t in pts], dtype=float)
                    y_arr = np.array([t[1] for t in pts], dtype=float)
                    mx, my = x_arr.mean(), y_arr.mean()
                    denom = ((x_arr - mx) ** 2).sum() or 1.0
                    slope = float(((x_arr - mx) * (y_arr - my)).sum() / denom)
                    intercept = float(my - slope * mx)
                    n = len(historique)
                    predictions[kpi] = {
                        'm1': round(slope * n + intercept, 2),
                        'm2': round(slope * (n + 1) + intercept, 2),
                        'm3': round(slope * (n + 2) + intercept, 2)
                    }

        return jsonify({
            'success': True,
            'kpis_suivi': kpis_noms,
            'historique': historique,
            'averages': averages,
            'predictions': predictions,
            'entreprise': current_user.entreprise or current_user.email
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# ÉVOLUTION — PREVIEW COLONNES
# ─────────────────────────────────────────
@app.route('/evolution_preview', methods=['POST'])
@login_required
def evolution_preview():
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'Aucun fichier reçu.'})
        df = lire_fichier(file)
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'Fichier vide.'})
        df.columns = [str(c).strip() for c in df.columns]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_col = detecter_date(df)
        kpis_existants = KpiSuivi.query.filter_by(user_id=current_user.id).order_by(KpiSuivi.ordre).all()
        kpis_noms = [k.nom for k in kpis_existants]
        return jsonify({
            'success': True,
            'all_cols': list(df.columns),
            'numeric_cols': numeric_cols,
            'date_col': date_col,
            'kpis_existants': kpis_noms
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# ÉVOLUTION — UPLOAD + SAUVEGARDE
# ─────────────────────────────────────────
@app.route('/evolution_upload', methods=['POST'])
@login_required
def evolution_upload():
    try:
        file = request.files.get('file')
        colonnes_select = request.form.getlist('colonnes')
        label_periode = request.form.get('label_periode', '').strip()
        sauvegarder_kpis = request.form.get('sauvegarder_kpis', 'true') == 'true'

        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'Aucun fichier reçu.'})
        if not colonnes_select:
            return jsonify({'success': False, 'error': 'Sélectionnez au moins un KPI.'})

        df = lire_fichier(file)
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'Fichier vide ou illisible.'})

        df.columns = [str(c).strip() for c in df.columns]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cols_valides = [c for c in colonnes_select if c in numeric_cols]

        if not cols_valides:
            return jsonify({'success': False, 'error': 'Aucun KPI numérique valide.'})

        if not label_periode:
            label_periode = datetime.utcnow().strftime('%B %Y')

        kpis_valeurs = {}
        for col in cols_valides:
            serie = df[col].dropna()
            if len(serie) > 0:
                kpis_valeurs[col] = round(float(serie.iloc[-1]), 2)

        periode_precedente = PeriodeEntreprise.query.filter_by(
            user_id=current_user.id
        ).order_by(PeriodeEntreprise.date_upload.desc()).first()

        kpis_precedents = {}
        if periode_precedente:
            valeurs_prec = ValeurKpi.query.filter_by(
                periode_id=periode_precedente.id,
                user_id=current_user.id
            ).all()
            kpis_precedents = {v.nom_kpi: v.valeur for v in valeurs_prec}

        score, note = calculer_score(kpis_valeurs, kpis_precedents)

        resume_ia = ''
        try:
            client_groq = groq.Groq(api_key=GROQ_API_KEY)
            kpis_str = ', '.join([f"{k}={v}" for k, v in kpis_valeurs.items()])
            prev_str = ', '.join([f"{k}={v}" for k, v in kpis_precedents.items()]) if kpis_precedents else 'première période'
            prompt = f"""Tu es consultant Boussole. Analyse la période pour un dirigeant de PME.
Note obtenue : {note} ({score}/100). KPIs actuels : {kpis_str}. Période précédente : {prev_str}.
Réponds UNIQUEMENT en JSON valide avec exactement ces 4 champs :
- "resume" : synthèse en 2 phrases pour le dirigeant
- "signal_fort" : le fait le plus marquant de la période, chiffré (1 phrase)
- "alerte" : un risque ou point de vigilance, ou null si tout va bien (1 phrase ou null)
- "conseil" : 1 action concrète et chiffrée à prendre ce mois-ci (1 phrase)
Exemple de format : {{"resume":"...","signal_fort":"...","alerte":null,"conseil":"..."}}"""
            resp = client_groq.chat.completions.create(
                model='llama-3.3-70b-versatile',
                response_format={"type": "json_object"},
                messages=[{'role': 'user', 'content': prompt}]
            )
            parsed_ia = json.loads(resp.choices[0].message.content)
            resume_ia = json.dumps({
                'resume': parsed_ia.get('resume', ''),
                'signal_fort': parsed_ia.get('signal_fort', None),
                'alerte': parsed_ia.get('alerte', None),
                'conseil': parsed_ia.get('conseil', None)
            }, ensure_ascii=False)
        except Exception:
            resume_ia = json.dumps({'resume': f"Période enregistrée avec une note {note}.", 'signal_fort': None, 'alerte': None, 'conseil': None}, ensure_ascii=False)

        nouvelle_periode = PeriodeEntreprise(
            user_id=current_user.id,
            date_upload=datetime.utcnow(),
            label_periode=label_periode,
            score=score,
            note=note,
            resume_ia=resume_ia
        )
        db.session.add(nouvelle_periode)
        db.session.flush()

        for nom, valeur in kpis_valeurs.items():
            val_kpi = ValeurKpi(
                periode_id=nouvelle_periode.id,
                user_id=current_user.id,
                nom_kpi=nom,
                valeur=valeur
            )
            db.session.add(val_kpi)

        if sauvegarder_kpis:
            KpiSuivi.query.filter_by(user_id=current_user.id).delete()
            for i, col in enumerate(cols_valides):
                kpi = KpiSuivi(user_id=current_user.id, nom=col, ordre=i)
                db.session.add(kpi)

        db.session.commit()

        toutes_periodes = PeriodeEntreprise.query.filter_by(
            user_id=current_user.id
        ).order_by(PeriodeEntreprise.date_upload).all()

        historique = []
        for p in toutes_periodes:
            valeurs = ValeurKpi.query.filter_by(periode_id=p.id, user_id=current_user.id).all()
            historique.append({
                'id': p.id,
                'date': p.date_upload.strftime('%d/%m/%Y'),
                'label': p.label_periode,
                'score': p.score,
                'note': p.note,
                'resume_ia': p.resume_ia,
                'valeurs': {v.nom_kpi: v.valeur for v in valeurs}
            })

        # Moyennes par KPI sur toutes les périodes
        averages = {}
        for kpi in cols_valides:
            vals = [p['valeurs'][kpi] for p in historique if kpi in p['valeurs'] and p['valeurs'][kpi] is not None]
            if vals:
                averages[kpi] = round(sum(vals) / len(vals), 2)

        # Prévisions M+1, M+2, M+3 par régression linéaire
        predictions = {}
        if len(historique) >= 2:
            for kpi in cols_valides:
                pts = [(i, p['valeurs'][kpi]) for i, p in enumerate(historique)
                       if kpi in p['valeurs'] and p['valeurs'][kpi] is not None]
                if len(pts) >= 2:
                    x_arr = np.array([t[0] for t in pts], dtype=float)
                    y_arr = np.array([t[1] for t in pts], dtype=float)
                    mx, my = x_arr.mean(), y_arr.mean()
                    denom = ((x_arr - mx) ** 2).sum() or 1.0
                    slope = float(((x_arr - mx) * (y_arr - my)).sum() / denom)
                    intercept = float(my - slope * mx)
                    n = len(historique)
                    predictions[kpi] = {
                        'm1': round(slope * n + intercept, 2),
                        'm2': round(slope * (n + 1) + intercept, 2),
                        'm3': round(slope * (n + 2) + intercept, 2)
                    }

        return jsonify({
            'success': True,
            'note': note,
            'score': score,
            'resume_ia': resume_ia,
            'label_periode': label_periode,
            'kpis_valeurs': kpis_valeurs,
            'historique': historique,
            'averages': averages,
            'predictions': predictions,
            'kpis_suivi': cols_valides,
            'cols_utilises': cols_valides,
            'nb_periodes': len(historique)
        })

    except Exception as e:
        db.session.rollback()
        print("ERREUR /evolution_upload :", str(e))
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# ÉVOLUTION — SUPPRIMER UNE PÉRIODE
# ─────────────────────────────────────────
@app.route('/evolution_supprimer/<int:periode_id>', methods=['DELETE'])
@login_required
def evolution_supprimer(periode_id):
    try:
        periode = PeriodeEntreprise.query.filter_by(
            id=periode_id, user_id=current_user.id
        ).first()
        if not periode:
            return jsonify({'success': False, 'error': 'Période introuvable.'})
        ValeurKpi.query.filter_by(periode_id=periode_id).delete()
        db.session.delete(periode)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────
# BENCHMARK SECTORIEL PME FRANÇAISES
# ─────────────────────────────────────────
@app.route('/benchmark', methods=['GET'])
@login_required
def benchmark():
    try:
        force = request.args.get('force', 'false') == 'true'
        if not force and session.get('benchmark_sectoriel'):
            return jsonify({'success': True, 'benchmark': session['benchmark_sectoriel'], 'cached': True})

        client_groq = groq.Groq(api_key=GROQ_API_KEY)
        prompt = """Tu es un expert en analyse financière et économique des PME françaises.
Tu dois générer un benchmark sectoriel complet pour une PME française.

CONTEXTE :
- Secteur : toutes PME françaises (10-250 salariés, CA entre 500k€ et 50M€)
- Source de référence : données INSEE, Banque de France (ratios sectoriels), BPI France
- Année de référence : 2023-2024

GÉNÈRE un benchmark JSON avec les moyennes sectorielles françaises pour ces indicateurs :

{
  "ca_et_marge": {
    "croissance_ca_annuelle": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "marge_brute": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "marge_nette": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "ebitda": {"moyenne": "X%", "bas": "X%", "haut": "X%"}
  },
  "couts_et_charges": {
    "charges_personnel_sur_ca": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "charges_fixes_sur_ca": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "cout_acquisition_client": {"moyenne": "X€", "bas": "X€", "haut": "X€"}
  },
  "clients_et_retention": {
    "taux_retention_client": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "taux_churn": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "panier_moyen": {"moyenne": "X€", "bas": "X€", "haut": "X€"},
    "nb_clients_actifs": {"moyenne": "X", "bas": "X", "haut": "X"}
  },
  "productivite_et_rh": {
    "ca_par_employe": {"moyenne": "X€", "bas": "X€", "haut": "X€"},
    "taux_absenteisme": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "turnover": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "masse_salariale_sur_ca": {"moyenne": "X%", "bas": "X%", "haut": "X%"}
  },
  "sante_financiere": {
    "ratio_liquidite": {"moyenne": "X", "bas": "X", "haut": "X"},
    "delai_paiement_clients": {"moyenne": "X jours", "bas": "X jours", "haut": "X jours"},
    "dette_sur_capital": {"moyenne": "X%", "bas": "X%", "haut": "X%"},
    "bfr_sur_ca": {"moyenne": "X%", "bas": "X%", "haut": "X%"}
  }
}

CONSIGNES :
- Utilise les vraies moyennes françaises issues des statistiques officielles
- "bas" = 25e percentile, "moyenne" = médiane, "haut" = 75e percentile
- Toutes les valeurs doivent être chiffrées et réalistes
- Réponds UNIQUEMENT en JSON valide sans texte avant ni après"""

        resp = client_groq.chat.completions.create(
            model='llama-3.3-70b-versatile',
            response_format={"type": "json_object"},
            messages=[{'role': 'user', 'content': prompt}]
        )
        data = json.loads(resp.choices[0].message.content)
        session['benchmark_sectoriel'] = data
        session.modified = True
        return jsonify({'success': True, 'benchmark': data, 'cached': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# ALERTES & RISQUES
# ─────────────────────────────────────────
@app.route('/alertes', methods=['GET'])
@login_required
def alertes():
    try:
        periodes = PeriodeEntreprise.query.filter_by(user_id=current_user.id).order_by(PeriodeEntreprise.date_upload.desc()).limit(2).all()
        if not periodes:
            return jsonify({'success': True, 'alertes': [], 'label_actuel': None, 'note': None, 'score': None})
        actuelle = periodes[0]
        precedente = periodes[1] if len(periodes) > 1 else None
        vals_act = {v.nom_kpi: v.valeur for v in ValeurKpi.query.filter_by(periode_id=actuelle.id, user_id=current_user.id).all()}
        vals_prec = {}
        if precedente:
            vals_prec = {v.nom_kpi: v.valeur for v in ValeurKpi.query.filter_by(periode_id=precedente.id, user_id=current_user.id).all()}
        alertes_detectees = []
        for kpi, val in vals_act.items():
            if kpi in vals_prec and vals_prec[kpi] != 0:
                delta_pct = (val - vals_prec[kpi]) / abs(vals_prec[kpi]) * 100
                if abs(delta_pct) >= 15:
                    niveau = 'critique' if abs(delta_pct) >= 25 else 'attention'
                    alertes_detectees.append({
                        'kpi': kpi, 'valeur': val, 'precedent': vals_prec[kpi],
                        'delta_pct': round(delta_pct, 1), 'niveau': niveau
                    })
        alertes_finales = []
        if alertes_detectees:
            try:
                client_groq = groq.Groq(api_key=GROQ_API_KEY)
                prompt = f"""Tu es consultant Boussole. Pour chaque alerte, génère un message court et une action concrète.
Alertes : {json.dumps(alertes_detectees, ensure_ascii=False)}
JSON : {{"alertes": [{{"kpi": "...", "message": "...", "action": "..."}}]}}"""
                resp = client_groq.chat.completions.create(
                    model='llama-3.3-70b-versatile',
                    response_format={"type": "json_object"},
                    messages=[{'role': 'user', 'content': prompt}]
                )
                ia_map = {a['kpi']: a for a in json.loads(resp.choices[0].message.content).get('alertes', [])}
                for a in alertes_detectees:
                    ia = ia_map.get(a['kpi'], {})
                    alertes_finales.append({**a, 'message': ia.get('message', ''), 'action': ia.get('action', '')})
            except Exception:
                alertes_finales = alertes_detectees
        else:
            alertes_finales = []
        return jsonify({
            'success': True, 'alertes': alertes_finales,
            'label_actuel': actuelle.label_periode,
            'note': actuelle.note, 'score': actuelle.score
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# WHAT-IF — CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
@app.route('/whatif_data', methods=['GET'])
@login_required
def whatif_data():
    try:
        derniere = PeriodeEntreprise.query.filter_by(user_id=current_user.id).order_by(PeriodeEntreprise.date_upload.desc()).first()
        if not derniere:
            return jsonify({'success': False, 'error': 'Aucune période disponible.'})

        vals_raw = {v.nom_kpi: v.valeur for v in ValeurKpi.query.filter_by(periode_id=derniere.id, user_id=current_user.id).all()}
        vals = {k: v for k, v in vals_raw.items() if not est_date_like(k) and v is not None}

        importances, source = resoudre_importances(current_user.id, list(vals.keys()))

        nb_periodes = PeriodeEntreprise.query.filter_by(user_id=current_user.id).count()
        source_labels = {
            'historique': f'Modèle RF/Pearson — {nb_periodes} périodes',
            'analyseur': 'Importances issues de l\'Analyseur (session courante)',
            'egal': 'Poids égaux — lancez l\'Analyseur ou ajoutez des périodes'
        }
        return jsonify({
            'success': True,
            'kpis': vals,
            'importances': importances,
            'importances_pct': {k: round(v*100, 1) for k, v in importances.items()},
            'label': derniere.label_periode,
            'score_base': derniere.score,
            'note_base': derniere.note,
            'has_model': source != 'egal',
            'source': source,
            'source_label': source_labels.get(source, ''),
            'nb_periodes': nb_periodes
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# WHAT-IF — SIMULATION PONDÉRÉE
# ─────────────────────────────────────────
@app.route('/whatif', methods=['POST'])
@login_required
def whatif():
    try:
        data = request.get_json()
        ajustements = data.get('ajustements', {})

        derniere = PeriodeEntreprise.query.filter_by(user_id=current_user.id).order_by(PeriodeEntreprise.date_upload.desc()).first()
        if not derniere:
            return jsonify({'success': False, 'error': 'Aucune donnée disponible.'})

        vals_raw = {v.nom_kpi: v.valeur for v in ValeurKpi.query.filter_by(periode_id=derniere.id, user_id=current_user.id).all()}
        vals_base = {k: v for k, v in vals_raw.items() if not est_date_like(k) and v is not None}

        # Valeurs simulées
        vals_sim = {kpi: round(val * (1 + ajustements.get(kpi, 0) / 100), 2) for kpi, val in vals_base.items()}

        importances, source = resoudre_importances(current_user.id, list(vals_base.keys()))

        # Score simulé pondéré : chaque % de variation est multiplié par l'importance du KPI
        score_base = derniere.score or 50.0
        delta_score = sum(
            importances.get(kpi, 0) * ajustements.get(kpi, 0) * 0.8
            for kpi in vals_base
        )
        score_simule = round(max(0.0, min(100.0, score_base + delta_score)), 1)

        if score_simule >= 80: note_sim = 'A'
        elif score_simule >= 65: note_sim = 'B'
        elif score_simule >= 45: note_sim = 'C'
        elif score_simule >= 25: note_sim = 'D'
        else: note_sim = 'E'

        importances_pct = {k: round(v*100, 1) for k, v in importances.items()}
        top3 = sorted(importances_pct.items(), key=lambda x: x[1], reverse=True)[:3]

        analyse_ia = impact = recommandation = ''
        try:
            client_groq = groq.Groq(api_key=GROQ_API_KEY)
            ajust_str = ', '.join([f"{k}:{v:+.0f}%" for k, v in ajustements.items() if v != 0]) or 'aucun ajustement'
            imp_str = ', '.join([f"{k}={p}%" for k, p in top3])
            prompt = f"""Tu es consultant Boussole. Analyse ce scénario what-if avec pondération par importance réelle des variables.
Top variables par influence sur le score : {imp_str}
Ajustements simulés : {ajust_str}
Score actuel : {score_base} ({derniere.note}) → Score simulé : {score_simule} ({note_sim})
JSON : {{"analyse": "3-4 phrases précises mentionnant les variables les plus influentes", "impact_principal": "1 phrase", "recommandation": "1 phrase concrète et chiffrée"}}"""
            resp = client_groq.chat.completions.create(
                model='llama-3.3-70b-versatile',
                response_format={"type": "json_object"},
                messages=[{'role': 'user', 'content': prompt}]
            )
            r = json.loads(resp.choices[0].message.content)
            analyse_ia = r.get('analyse', '')
            impact = r.get('impact_principal', '')
            recommandation = r.get('recommandation', '')
        except Exception:
            analyse_ia = f"Simulation pondérée : note {note_sim} ({score_simule}/100)."

        return jsonify({
            'success': True,
            'vals_base': vals_base,
            'vals_simules': vals_sim,
            'score_base': score_base,
            'note_base': derniere.note,
            'score_simule': score_simule,
            'note_simulee': note_sim,
            'importances_pct': importances_pct,
            'analyse_ia': analyse_ia,
            'impact': impact,
            'recommandation': recommandation
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# EXPORT PDF
# ─────────────────────────────────────────
@app.route('/export_pdf', methods=['POST'])
@login_required
def export_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.enums import TA_CENTER

        data = request.get_json() or {}
        derniere = PeriodeEntreprise.query.filter_by(user_id=current_user.id).order_by(PeriodeEntreprise.date_upload.desc()).first()
        vals_kpis = {}
        if derniere:
            vals_kpis = {v.nom_kpi: v.valeur for v in ValeurKpi.query.filter_by(periode_id=derniere.id, user_id=current_user.id).all()}

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        PRIMARY = colors.HexColor('#4338ca')

        s_title = ParagraphStyle('TC', parent=styles['Title'], textColor=PRIMARY, fontSize=22, spaceAfter=4)
        s_sub = ParagraphStyle('TS', parent=styles['Normal'], textColor=colors.HexColor('#64748b'), fontSize=10, spaceAfter=16)
        s_h2 = ParagraphStyle('TH2', parent=styles['Heading2'], textColor=PRIMARY, fontSize=13, spaceBefore=12, spaceAfter=6)
        s_body = ParagraphStyle('TB', parent=styles['Normal'], fontSize=10, leading=16, spaceAfter=8)
        s_foot = ParagraphStyle('TF', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#94a3b8'), alignment=TA_CENTER)

        story = []
        story.append(Paragraph("Boussole", s_title))
        story.append(Paragraph(f"Synthèse Exécutive — {datetime.utcnow().strftime('%d/%m/%Y')}", s_sub))
        entreprise = current_user.entreprise or current_user.email
        if derniere:
            story.append(Paragraph(f"Entreprise : <b>{entreprise}</b> · Période : <b>{derniere.label_periode}</b> · Note santé : <b>{derniere.note}</b> ({derniere.score}/100)", s_body))
        story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=12))

        if vals_kpis:
            story.append(Paragraph("KPIs Clés", s_h2))
            tbl_data = [['Indicateur', 'Valeur']] + [[k, f"{v:,.2f}".replace(',', ' ')] for k, v in vals_kpis.items()]
            t = Table(tbl_data, colWidths=[10*cm, 5*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.4*cm))

        if derniere and derniere.resume_ia:
            story.append(Paragraph("Synthèse de Période", s_h2))
            story.append(Paragraph(derniere.resume_ia, s_body))

        analyse = data.get('analyse', '').strip()
        if analyse:
            story.append(Paragraph("Analyse & Notes", s_h2))
            story.append(Paragraph(analyse.replace('\n', '<br/>'), s_body))

        recos = data.get('recommandations', [])
        if recos:
            story.append(Paragraph("Recommandations Prioritaires", s_h2))
            for i, rec in enumerate(recos[:3], 1):
                story.append(Paragraph(f"<b>{i}. {rec.get('action', '')}</b>", s_body))
                if rec.get('detail'):
                    story.append(Paragraph(rec['detail'], s_body))

        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1')))
        story.append(Paragraph(f"Document généré par Boussole · {datetime.utcnow().strftime('%d/%m/%Y à %H:%M')}", s_foot))

        doc.build(story)
        buffer.seek(0)
        filename = f"synthese_executive_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf"
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')

    except ImportError:
        return jsonify({'success': False, 'error': 'ReportLab non installé. Exécutez : pip install reportlab'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# INIT DB + LANCEMENT
# ─────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)