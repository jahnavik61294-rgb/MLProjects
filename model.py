from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb

def build_models(preprocessor):
    """Build multiple models for comparison - optimized for speed"""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1),
        'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=500, n_jobs=-1),
        #'NaiveBayes': MultinomialNB(),  # Works with sparse data, no memory issues
        'GradientBoosting': GradientBoostingClassifier(n_estimators=50, max_depth=6, random_state=42),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(n_estimators=50, max_depth=6, random_state=42, eval_metric='logloss', n_jobs=-1)
    }
    
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    return pipelines