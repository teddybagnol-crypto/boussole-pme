import os
import io
import json
import zipfile
import tempfile
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

class UserProfile(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    nom_complet   = db.Column(db.String(200), default='')
    nom_entreprise= db.Column(db.String(200), default='')
    secteur       = db.Column(db.String(100), default='')
    nb_employes   = db.Column(db.String(50),  default='')

class UserPreferences(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    theme            = db.Column(db.String(20),  default='light')
    couleur_accent   = db.Column(db.String(7),   default='#2563EB')
    langue           = db.Column(db.String(5),   default='fr')
    format_date      = db.Column(db.String(20),  default='DD/MM/YYYY')
    devise           = db.Column(db.String(5),   default='EUR')
    notif_analyse    = db.Column(db.Boolean,     default=True)
    notif_score      = db.Column(db.Boolean,     default=True)
    notif_rapport    = db.Column(db.Boolean,     default=False)
    freq_resume      = db.Column(db.String(20),  default='hebdomadaire')
    profil_public    = db.Column(db.Boolean,     default=False)


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

_MOIS_MAP = {
    'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3, 'avril': 4,
    'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8, 'aout': 8,
    'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12, 'decembre': 12,
    'jan': 1, 'fév': 2, 'fev': 2, 'mar': 3, 'avr': 4,
    'jun': 6, 'jul': 7, 'aoû': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'déc': 12, 'dec': 12,
}

def _parse_periode(val):
    """Parse une valeur en datetime — gère les formats français et standard."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    # "Janvier 2024", "Janvier-2024", "janv. 2024"
    for sep in (' ', '-', '/'):
        parts = [p.strip().rstrip('.') for p in s.split(sep)]
        if len(parts) == 2:
            a, b = parts
            if a in _MOIS_MAP and b.isdigit() and len(b) == 4:
                return datetime(int(b), _MOIS_MAP[a], 1)
            if b in _MOIS_MAP and a.isdigit() and len(a) == 4:
                return datetime(int(a), _MOIS_MAP[b], 1)
    # "01/2024", "2024-01"
    for fmt in ('%m/%Y', '%Y-%m', '%m-%Y', '%Y/%m'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    # Parseur pandas en dernier recours
    try:
        return pd.to_datetime(val, dayfirst=True).to_pydatetime()
    except Exception:
        return None

def detecter_date(df):
    """Retourne le nom de la colonne la plus probable comme axe temporel."""
    for c in df.columns:
        try:
            parsed = df[c].apply(_parse_periode)
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
    if current_user.is_authenticated:
        return redirect('/dashboard')
    return render_template('login.html')

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
# ÉVOLUTION — HELPERS
# ─────────────────────────────────────────

_MOIS_FR = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
            'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']

def _label_from_date(dt):
    return f"{_MOIS_FR[dt.month - 1]} {dt.year}"


def _generer_resume_ia(note, score, kpis_valeurs, kpis_precedents):
    kpis_str = ', '.join(f"{k}={v}" for k, v in kpis_valeurs.items())
    prev_str = (', '.join(f"{k}={v}" for k, v in kpis_precedents.items())
                if kpis_precedents else 'première période')
    try:
        client_groq = groq.Groq(api_key=GROQ_API_KEY)
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
        parsed = json.loads(resp.choices[0].message.content)
        return json.dumps({
            'resume': parsed.get('resume', ''),
            'signal_fort': parsed.get('signal_fort', None),
            'alerte': parsed.get('alerte', None),
            'conseil': parsed.get('conseil', None)
        }, ensure_ascii=False)
    except Exception:
        return json.dumps({
            'resume': f"Période enregistrée avec une note {note} ({score}/100).",
            'signal_fort': None, 'alerte': None, 'conseil': None
        }, ensure_ascii=False)


# ─────────────────────────────────────────
# ÉVOLUTION — UPLOAD + SAUVEGARDE
# ─────────────────────────────────────────
@app.route('/evolution_upload', methods=['POST'])
@login_required
def evolution_upload():
    try:
        file = request.files.get('file')
        colonnes_select = request.form.getlist('colonnes')
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

        # ── 1. Index des périodes existantes (nécessaire avant de construire lignes) ─
        periodes_par_label = {
            p.label_periode: p
            for p in PeriodeEntreprise.query.filter_by(user_id=current_user.id).all()
        }
        existing_labels = set(periodes_par_label.keys())

        # ── 2. Construire la liste de lignes (date, label, kpis) ────────
        date_col = detecter_date(df)
        lignes = []  # [(datetime, label_str, {kpi: val})]

        # Exclure la colonne date des KPIs (peut être lue comme float en Excel)
        if date_col and date_col in cols_valides:
            cols_valides = [c for c in cols_valides if c != date_col]
        if not cols_valides:
            return jsonify({'success': False, 'error': 'Aucun KPI numérique valide (hors colonne date).'})

        if date_col:
            # Parser flexible : gère "Janvier 2024", "01/2024", dates standard…
            df['_dt'] = df[date_col].apply(_parse_periode)
            df_sorted = df.dropna(subset=['_dt']).sort_values('_dt').reset_index(drop=True)
            for _, row in df_sorted.iterrows():
                dt = row['_dt']
                if not isinstance(dt, datetime):
                    dt = dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else datetime(dt.year, dt.month, dt.day)
                label = _label_from_date(dt)
                kpis = {}
                for col in cols_valides:
                    val = row.get(col)
                    if val is not None and pd.notna(val) and not isinstance(val, (pd.Timestamp, datetime)):
                        try:
                            kpis[col] = round(float(val), 2)
                        except (TypeError, ValueError):
                            pass
                if kpis:
                    lignes.append((dt, label, kpis))
            df.drop(columns=['_dt'], inplace=True, errors='ignore')
            # Dédoublonner par label : même mois → garder la dernière ligne
            seen: dict = {}
            for entry in lignes:
                seen[entry[1]] = entry
            lignes = list(seen.values())
        else:
            # Pas de colonne date : une période par upload avec label unique
            dt = datetime.utcnow()
            base_label = _label_from_date(dt)
            label = base_label
            counter = 2
            while label in existing_labels:
                label = f"{base_label} ({counter})"
                counter += 1
            kpis = {col: round(float(df[col].dropna().iloc[-1]), 2)
                    for col in cols_valides if len(df[col].dropna()) > 0}
            if kpis:
                lignes.append((dt, label, kpis))

        if not lignes:
            return jsonify({'success': False, 'error': 'Aucune donnée valide trouvée.'})

        # ── 3. KPIs de référence : période DB antérieure au premier import ─
        premiere_date = lignes[0][0]
        periode_ref = PeriodeEntreprise.query.filter(
            PeriodeEntreprise.user_id == current_user.id,
            PeriodeEntreprise.date_upload < premiere_date
        ).order_by(PeriodeEntreprise.date_upload.desc()).first()

        kpis_prev: dict = {}
        if periode_ref:
            kpis_prev = {v.nom_kpi: v.valeur for v in
                         ValeurKpi.query.filter_by(
                             periode_id=periode_ref.id,
                             user_id=current_user.id).all()}

        derniere_periode = None

        for i, (dt, label, kpis_valeurs) in enumerate(lignes):
            score, note = calculer_score(kpis_valeurs, kpis_prev or None)
            is_last = (i == len(lignes) - 1)

            # Résumé IA uniquement pour la dernière période (évite N appels API)
            if is_last:
                resume_ia = _generer_resume_ia(note, score, kpis_valeurs, kpis_prev)
            else:
                resume_ia = json.dumps({
                    'resume': f"Période {label} — note {note} ({score}/100).",
                    'signal_fort': None, 'alerte': None, 'conseil': None
                }, ensure_ascii=False)

            # UPSERT — update si la période existe déjà, insert sinon
            if label in periodes_par_label:
                p = periodes_par_label[label]
                p.score = score
                p.note = note
                p.resume_ia = resume_ia
                p.date_upload = dt
                ValeurKpi.query.filter_by(
                    periode_id=p.id, user_id=current_user.id).delete()
            else:
                p = PeriodeEntreprise(
                    user_id=current_user.id,
                    date_upload=dt,
                    label_periode=label,
                    score=score,
                    note=note,
                    resume_ia=resume_ia
                )
                db.session.add(p)
                db.session.flush()
                periodes_par_label[label] = p

            for nom, valeur in kpis_valeurs.items():
                db.session.add(ValeurKpi(
                    periode_id=p.id,
                    user_id=current_user.id,
                    nom_kpi=nom,
                    valeur=valeur
                ))

            kpis_prev = kpis_valeurs   # référence pour la prochaine itération
            derniere_periode = p

        # ── 4. KPIs suivis ───────────────────────────────────────────────
        if sauvegarder_kpis:
            KpiSuivi.query.filter_by(user_id=current_user.id).delete()
            for i, col in enumerate(cols_valides):
                db.session.add(KpiSuivi(user_id=current_user.id, nom=col, ordre=i))

        db.session.commit()

        # ── 5. Réponse — historique complet + statistiques ───────────────
        toutes_periodes = PeriodeEntreprise.query.filter_by(
            user_id=current_user.id
        ).order_by(PeriodeEntreprise.date_upload).all()

        historique = []
        for p in toutes_periodes:
            valeurs = ValeurKpi.query.filter_by(
                periode_id=p.id, user_id=current_user.id).all()
            historique.append({
                'id': p.id,
                'date': p.date_upload.strftime('%d/%m/%Y'),
                'label': p.label_periode,
                'score': p.score,
                'note': p.note,
                'resume_ia': p.resume_ia,
                'valeurs': {v.nom_kpi: v.valeur for v in valeurs}
            })

        averages = {}
        for kpi in cols_valides:
            vals = [p['valeurs'][kpi] for p in historique
                    if kpi in p['valeurs'] and p['valeurs'][kpi] is not None]
            if vals:
                averages[kpi] = round(sum(vals) / len(vals), 2)

        predictions = {}
        if len(historique) >= 2:
            for kpi in cols_valides:
                pts = [(i, p['valeurs'][kpi]) for i, p in enumerate(historique)
                       if kpi in p['valeurs'] and p['valeurs'][kpi] is not None]
                if len(pts) >= 2:
                    xs = np.array([t[0] for t in pts], dtype=float)
                    ys = np.array([t[1] for t in pts], dtype=float)
                    mx, my = xs.mean(), ys.mean()
                    denom = ((xs - mx) ** 2).sum() or 1.0
                    slope = float(((xs - mx) * (ys - my)).sum() / denom)
                    intercept = float(my - slope * mx)
                    n = len(historique)
                    predictions[kpi] = {
                        'm1': round(slope * n + intercept, 2),
                        'm2': round(slope * (n + 1) + intercept, 2),
                        'm3': round(slope * (n + 2) + intercept, 2)
                    }

        dp = derniere_periode
        return jsonify({
            'success': True,
            'note': dp.note,
            'score': dp.score,
            'resume_ia': dp.resume_ia,
            'label_periode': dp.label_periode,
            'kpis_valeurs': lignes[-1][2],
            'historique': historique,
            'averages': averages,
            'predictions': predictions,
            'kpis_suivi': cols_valides,
            'cols_utilises': cols_valides,
            'nb_periodes': len(historique),
            'nb_periodes_importees': len(lignes)
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
# BENCHMARK — QUESTION IA
# ─────────────────────────────────────────
@app.route('/benchmark_question', methods=['POST'])
@login_required
def benchmark_question():
    try:
        data = request.get_json()
        question = (data.get('question') or '').strip()
        valeurs_client = data.get('valeurs_client', {})

        if not question:
            return jsonify({'success': False, 'error': 'Question vide.'})

        bench = session.get('benchmark_sectoriel')
        if not bench:
            return jsonify({'success': False, 'error': 'Générez d\'abord le benchmark (cliquez sur "Comparer au secteur").'})

        # Score santé depuis la dernière période enregistrée
        score_info = ''
        derniere = PeriodeEntreprise.query.filter_by(
            user_id=current_user.id
        ).order_by(PeriodeEntreprise.date_upload.desc()).first()
        if derniere:
            score_info = f"Score de santé global Boussole : {derniere.note} ({derniere.score}/100) — {derniere.label_periode}\n"

        # Construction du contexte benchmark + valeurs client
        cat_names = {
            'ca_et_marge':          'CA & Marges',
            'couts_et_charges':     'Coûts & Charges',
            'clients_et_retention': 'Clients & Rétention',
            'productivite_et_rh':   'Productivité & RH',
            'sante_financiere':     'Santé Financière'
        }
        ctx_lines = []
        for cat_key, cat_label in cat_names.items():
            cat_data = bench.get(cat_key, {})
            if not cat_data:
                continue
            ctx_lines.append(f"\n[{cat_label}]")
            for met_key, vals in cat_data.items():
                user_val = valeurs_client.get(f"{cat_key}__{met_key}")
                user_str = f" | CLIENT={user_val}" if user_val is not None else " | CLIENT=non renseigné"
                ctx_lines.append(
                    f"  {met_key}: P25={vals.get('bas','?')} · médiane={vals.get('moyenne','?')} · P75={vals.get('haut','?')}{user_str}"
                )

        context_str = score_info + '\n'.join(ctx_lines)

        # Historique conversation (2 derniers échanges pour le contexte)
        historique_conv = session.get('benchmark_conv', [])
        hist_str = ''
        if historique_conv:
            hist_str = '\n'.join([
                f"Q: {h['question']}\nR: {h.get('reponse_directe', '')[:250]}"
                for h in historique_conv[-2:]
            ])

        system_prompt = """Tu es conseiller stratégique Boussole, expert en performance des PME françaises (10-250 salariés, CA 500k€-50M€). Tu analyses la position d'un dirigeant face aux benchmarks sectoriels INSEE / Banque de France / BPI France.

RÈGLES ABSOLUES :
1. Chiffrer chaque affirmation : écart en %, montants en €, délais en jours
2. Calculer l'écart précis client vs médiane si la valeur CLIENT est fournie
3. Actions concrètes : inclure délai réaliste et impact chiffré estimé
4. Si la question ne concerne pas la performance / benchmark PME : répondre poliment dans "reponse_directe" et mettre "non applicable" dans "position_marche"
5. Répondre UNIQUEMENT en JSON valide — aucun texte avant ni après

FORMAT OBLIGATOIRE :
{
  "reponse_directe": "réponse précise et chiffrée à la question (2-3 phrases)",
  "position_marche": "en dessous de la médiane | dans la médiane | au dessus de la médiane | non applicable",
  "ecart_chiffre": "X% d'écart avec la médiane sectorielle (ou 'Non calculable' si valeur client absente)",
  "causes_probables": ["cause 1 concrète et chiffrée", "cause 2 concrète et chiffrée"],
  "actions_concretes": [
    {"action": "Action précise et actionnable", "impact_estime": "impact chiffré attendu", "effort": "Faible"},
    {"action": "...", "impact_estime": "...", "effort": "Moyen"},
    {"action": "...", "impact_estime": "...", "effort": "Fort"}
  ],
  "comparaison_visuelle": {
    "label": "Nom de l'indicateur principal concerné par la question",
    "unite": "% ou € ou jours ou vide",
    "client": null,
    "moyenne_secteur": null,
    "top25_secteur": null
  }
}"""

        user_prompt = f"""DONNÉES BENCHMARK PME FRANÇAISES + VALEURS DU CLIENT :
{context_str}
{f"{chr(10)}HISTORIQUE (pour questions de suivi) :{chr(10)}{hist_str}" if hist_str else ""}

QUESTION DU DIRIGEANT : {question}"""

        client_groq = groq.Groq(api_key=GROQ_API_KEY)
        resp = client_groq.chat.completions.create(
            model='llama-3.3-70b-versatile',
            response_format={"type": "json_object"},
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_prompt}
            ],
            temperature=0.2
        )
        result = json.loads(resp.choices[0].message.content)

        # Sauvegarder en session (garder 3 derniers échanges)
        historique_conv.append({
            'question':      question,
            'reponse_directe': result.get('reponse_directe', ''),
            'position_marche': result.get('position_marche', ''),
        })
        session['benchmark_conv'] = historique_conv[-3:]
        session.modified = True

        return jsonify({'success': True, **result, 'historique': session['benchmark_conv']})

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
# PARAMÈTRES
# ─────────────────────────────────────────
def _get_or_create_profile(user_id):
    p = UserProfile.query.filter_by(user_id=user_id).first()
    if not p:
        p = UserProfile(user_id=user_id)
        db.session.add(p)
        db.session.flush()
    return p

def _get_or_create_prefs(user_id):
    p = UserPreferences.query.filter_by(user_id=user_id).first()
    if not p:
        p = UserPreferences(user_id=user_id)
        db.session.add(p)
        db.session.flush()
    return p

@app.route('/parametres')
@login_required
def parametres():
    profile = _get_or_create_profile(current_user.id)
    prefs   = _get_or_create_prefs(current_user.id)
    db.session.commit()
    kpis = KpiSuivi.query.filter_by(user_id=current_user.id).order_by(KpiSuivi.ordre).all()
    periodes = PeriodeEntreprise.query.filter_by(user_id=current_user.id)\
               .order_by(PeriodeEntreprise.date_upload.desc()).all()
    nb_valeurs = ValeurKpi.query.filter_by(user_id=current_user.id).count()
    return render_template('parametres.html',
                           user=current_user, profile=profile, prefs=prefs,
                           kpis=kpis, periodes=periodes, nb_valeurs=nb_valeurs)

@app.route('/parametres/profil', methods=['POST'])
@login_required
def parametres_profil():
    try:
        data = request.get_json()
        profile = _get_or_create_profile(current_user.id)
        profile.nom_complet    = (data.get('nom_complet') or '').strip()[:200]
        profile.nom_entreprise = (data.get('nom_entreprise') or '').strip()[:200]
        profile.secteur        = (data.get('secteur') or '').strip()[:100]
        profile.nb_employes    = (data.get('nb_employes') or '').strip()[:50]
        email = (data.get('email') or '').strip().lower()
        if email and email != current_user.email:
            if User.query.filter_by(email=email).first():
                return jsonify({'success': False, 'error': 'Cet email est déjà utilisé.'})
            current_user.email = email
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parametres/securite', methods=['POST'])
@login_required
def parametres_securite():
    try:
        data = request.get_json()
        ancien = data.get('ancien_mdp', '')
        nouveau = data.get('nouveau_mdp', '')
        confirm = data.get('confirm_mdp', '')
        if not check_password_hash(current_user.password, ancien):
            return jsonify({'success': False, 'error': 'Mot de passe actuel incorrect.'})
        if len(nouveau) < 8:
            return jsonify({'success': False, 'error': 'Le nouveau mot de passe doit faire au moins 8 caractères.'})
        if nouveau != confirm:
            return jsonify({'success': False, 'error': 'Les deux mots de passe ne correspondent pas.'})
        current_user.password = generate_password_hash(nouveau)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parametres/preferences', methods=['POST'])
@login_required
def parametres_preferences():
    try:
        data  = request.get_json()
        prefs = _get_or_create_prefs(current_user.id)
        prefs.theme          = data.get('theme', prefs.theme)
        prefs.couleur_accent = data.get('couleur_accent', prefs.couleur_accent)
        prefs.langue         = data.get('langue', prefs.langue)
        prefs.format_date    = data.get('format_date', prefs.format_date)
        prefs.devise         = data.get('devise', prefs.devise)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parametres/notifications', methods=['POST'])
@login_required
def parametres_notifications():
    try:
        data  = request.get_json()
        prefs = _get_or_create_prefs(current_user.id)
        prefs.notif_analyse = bool(data.get('notif_analyse', prefs.notif_analyse))
        prefs.notif_score   = bool(data.get('notif_score',   prefs.notif_score))
        prefs.notif_rapport = bool(data.get('notif_rapport', prefs.notif_rapport))
        prefs.freq_resume   = data.get('freq_resume', prefs.freq_resume)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parametres/kpis/renommer', methods=['POST'])
@login_required
def parametres_kpi_renommer():
    try:
        data = request.get_json()
        kpi  = KpiSuivi.query.filter_by(id=data.get('id'), user_id=current_user.id).first()
        if not kpi:
            return jsonify({'success': False, 'error': 'KPI introuvable.'})
        kpi.nom = (data.get('nom') or '').strip()[:150]
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parametres/kpis/<int:kpi_id>', methods=['DELETE'])
@login_required
def parametres_kpi_supprimer(kpi_id):
    try:
        kpi = KpiSuivi.query.filter_by(id=kpi_id, user_id=current_user.id).first()
        if not kpi:
            return jsonify({'success': False, 'error': 'KPI introuvable.'})
        db.session.delete(kpi)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parametres/kpis/ordre', methods=['POST'])
@login_required
def parametres_kpi_ordre():
    try:
        data  = request.get_json()
        ordre = data.get('ordre', [])
        for i, kpi_id in enumerate(ordre):
            kpi = KpiSuivi.query.filter_by(id=kpi_id, user_id=current_user.id).first()
            if kpi:
                kpi.ordre = i
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parametres/export_donnees')
@login_required
def parametres_export():
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            # profil.json
            profile = _get_or_create_profile(current_user.id)
            prefs   = _get_or_create_prefs(current_user.id)
            profil_data = {
                'email': current_user.email,
                'entreprise': current_user.entreprise,
                'nom_complet': profile.nom_complet,
                'nom_entreprise': profile.nom_entreprise,
                'secteur': profile.secteur,
                'nb_employes': profile.nb_employes,
            }
            zf.writestr('profil.json', json.dumps(profil_data, ensure_ascii=False, indent=2))

            # periodes.csv
            periodes = PeriodeEntreprise.query.filter_by(user_id=current_user.id).all()
            if periodes:
                rows = [['label_periode','date_upload','score','note','resume_ia']]
                for p in periodes:
                    rows.append([p.label_periode,
                                 p.date_upload.strftime('%Y-%m-%d') if p.date_upload else '',
                                 p.score, p.note, (p.resume_ia or '').replace('\n',' ')])
                csv_txt = '\n'.join(','.join(f'"{str(c)}"' for c in r) for r in rows)
                zf.writestr('periodes.csv', csv_txt)

            # kpis_valeurs.csv
            valeurs = ValeurKpi.query.filter_by(user_id=current_user.id).all()
            if valeurs:
                rows = [['periode_id','nom_kpi','valeur']]
                for v in valeurs:
                    rows.append([v.periode_id, v.nom_kpi, v.valeur])
                csv_txt = '\n'.join(','.join(f'"{str(c)}"' for c in r) for r in rows)
                zf.writestr('kpis_valeurs.csv', csv_txt)

            # preferences.json
            prefs_data = {
                'theme': prefs.theme, 'couleur_accent': prefs.couleur_accent,
                'langue': prefs.langue, 'format_date': prefs.format_date,
                'devise': prefs.devise,
            }
            zf.writestr('preferences.json', json.dumps(prefs_data, ensure_ascii=False, indent=2))

        buf.seek(0)
        fname = f"boussole_export_{datetime.utcnow().strftime('%Y%m%d')}.zip"
        return send_file(buf, as_attachment=True, download_name=fname,
                         mimetype='application/zip')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parametres/supprimer_historique', methods=['POST'])
@login_required
def parametres_supprimer_historique():
    try:
        ValeurKpi.query.filter_by(user_id=current_user.id).delete()
        PeriodeEntreprise.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parametres/supprimer_compte', methods=['POST'])
@login_required
def parametres_supprimer_compte():
    try:
        uid = current_user.id
        logout_user()
        ValeurKpi.query.filter_by(user_id=uid).delete()
        PeriodeEntreprise.query.filter_by(user_id=uid).delete()
        KpiSuivi.query.filter_by(user_id=uid).delete()
        UserProfile.query.filter_by(user_id=uid).delete()
        UserPreferences.query.filter_by(user_id=uid).delete()
        User.query.filter_by(id=uid).delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# COMMUNAUTÉ
# ─────────────────────────────────────────

@app.route('/communaute')
@login_required
def communaute():
    return render_template('communaute.html', user=current_user)


@app.route('/api/communaute/feed')
@login_required
def communaute_feed():
    """Retourne le feed anonymisé de toutes les entreprises actives."""
    try:
        all_users = User.query.all()
        cards = []
        for u in all_users:
            periodes = PeriodeEntreprise.query.filter_by(user_id=u.id).order_by(PeriodeEntreprise.date_upload.desc()).all()
            if not periodes:
                continue
            profile = UserProfile.query.filter_by(user_id=u.id).first()
            prefs   = UserPreferences.query.filter_by(user_id=u.id).first()
            is_public = (prefs and prefs.profil_public)
            secteur     = (profile.secteur    if profile and profile.secteur    else 'Non renseigné')
            nb_employes = (profile.nb_employes if profile and profile.nb_employes else '—')
            nom_entreprise = (profile.nom_entreprise if profile and profile.nom_entreprise else '') if is_public else ''

            derniere = periodes[0]
            note = derniere.note or '—'
            score = round(derniere.score, 1) if derniere.score else None

            # Evolution entre les 2 dernières périodes
            evolution = None
            if len(periodes) >= 2 and periodes[0].score and periodes[1].score:
                evolution = round(periodes[0].score - periodes[1].score, 1)

            nb_kpis = KpiSuivi.query.filter_by(user_id=u.id).count()
            nb_periodes = len(periodes)
            derniere_activite = derniere.date_upload.strftime('%d/%m/%Y') if derniere.date_upload else '—'

            is_me = (u.id == current_user.id)

            cards.append({
                'id':               u.id,
                'is_me':            is_me,
                'is_public':        is_public,
                'nom_entreprise':   nom_entreprise,
                'secteur':          secteur,
                'nb_employes':      nb_employes,
                'note':             note,
                'score':            score,
                'evolution':        evolution,
                'nb_kpis':          nb_kpis,
                'nb_periodes':      nb_periodes,
                'derniere_activite': derniere_activite,
            })

        # Ma carte en premier, puis tri par score desc
        cards.sort(key=lambda c: (0 if c['is_me'] else 1, -(c['score'] or 0)))
        return jsonify({'success': True, 'cards': cards, 'total': len(cards)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/communaute/opt-in', methods=['POST'])
@login_required
def communaute_opt_in():
    """Toggle la visibilité publique du profil."""
    try:
        prefs = UserPreferences.query.filter_by(user_id=current_user.id).first()
        if not prefs:
            prefs = UserPreferences(user_id=current_user.id)
            db.session.add(prefs)
        prefs.profil_public = not prefs.profil_public
        db.session.commit()
        return jsonify({'success': True, 'profil_public': prefs.profil_public})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────
# INIT DB + LANCEMENT
# ─────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)