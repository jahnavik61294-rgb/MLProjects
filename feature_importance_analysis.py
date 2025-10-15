import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split
from preprocessing import build_preprocessor
from model import build_models
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path):
    """Load and prepare data for feature importance analysis"""
    df = pd.read_csv(data_path, low_memory=False)
    
    # Data cleaning (same as training pipeline)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).fillna('Unknown')
    
    mixed_cols = [1,7,8,16,17,18,19,20,35]
    for c in mixed_cols:
        df.iloc[:, c] = df.iloc[:, c].astype(str).fillna('Unknown')
    
    return df

def get_feature_names(preprocessor, X_train):
    """Extract feature names after preprocessing"""
    # Fit preprocessor to get feature names
    X_transformed = preprocessor.fit_transform(X_train)
    
    # Get feature names from column transformer
    feature_names = []
    
    # Numerical features
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    feature_names.extend([f"num_{name}" for name in num_features])
    
    # Categorical features
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    feature_names.extend([f"cat_{name}" for name in cat_features])
    
    return feature_names

def analyze_model_importance(model, model_name, X_train, y_train, X_test, feature_names):
    """Analyze feature importance for a single model"""
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS: {model_name}")
    print(f"{'='*60}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 1. Built-in Feature Importance (for tree-based models)
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features ({model_name}):")
        print("-" * 50)
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:<30} {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    # 2. SHAP Analysis (for all models)
    print(f"\nGenerating SHAP analysis for {model_name}...")
    
    # Sample data for SHAP (to avoid memory issues)
    sample_size = min(1000, len(X_test))
    X_sample = X_test[:sample_size]
    
    try:
        # Create SHAP explainer
        if model_name in ['LogisticRegression']:
            # For linear models, use LinearExplainer
            explainer = shap.LinearExplainer(model.named_steps['model'], X_train)
        else:
            # For tree-based models, use TreeExplainer
            explainer = shap.TreeExplainer(model.named_steps['model'])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification
        if len(shap_values) == 2:
            shap_values = shap_values[1]  # Use positive class
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create SHAP bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_bar_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features (SHAP - {model_name}):")
        print("-" * 50)
        for i, row in shap_importance.head(10).iterrows():
            print(f"{row['feature']:<30} {row['shap_importance']:.4f}")
        
        return shap_importance
        
    except Exception as e:
        print(f"SHAP analysis failed for {model_name}: {str(e)}")
        return None

def create_combined_importance_plot(all_importances, model_names):
    """Create a combined feature importance plot for all models"""
    plt.figure(figsize=(15, 10))
    
    # Get top 10 features from each model
    top_features = set()
    for importance_df in all_importances.values():
        if importance_df is not None:
            top_features.update(importance_df.head(10)['feature'].tolist())
    
    # Create a matrix of importances
    importance_matrix = []
    feature_list = list(top_features)[:15]  # Limit to top 15
    
    for model_name in model_names:
        if model_name in all_importances and all_importances[model_name] is not None:
            model_importance = all_importances[model_name]
            model_scores = []
            for feature in feature_list:
                if feature in model_importance['feature'].values:
                    score = model_importance[model_importance['feature'] == feature]['shap_importance'].iloc[0]
                else:
                    score = 0
                model_scores.append(score)
            importance_matrix.append(model_scores)
        else:
            importance_matrix.append([0] * len(feature_list))
    
    # Normalize scores
    importance_matrix = np.array(importance_matrix)
    importance_matrix = importance_matrix / (importance_matrix.max(axis=1, keepdims=True) + 1e-8)
    
    # Create heatmap
    sns.heatmap(importance_matrix, 
                xticklabels=feature_list,
                yticklabels=model_names,
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Importance'})
    
    plt.title('Feature Importance Comparison Across All Models', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('combined_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main feature importance analysis"""
    print("Starting Feature Importance Analysis...")
    
    # Load data
    df = load_and_prepare_data('Data/Dataset.csv')
    
    # Prepare features and target
    target = 'Default'
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # Build preprocessor
    num = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat = X.select_dtypes(include=['object']).columns.tolist()
    preprocessor = build_preprocessor(num, cat, [])
    
    # Get feature names
    feature_names = get_feature_names(preprocessor, X_train)
    print(f"Total features after preprocessing: {len(feature_names)}")
    
    # Get models
    models = build_models(preprocessor)
    
    # Analyze each model
    all_importances = {}
    model_names = []
    
    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name}...")
        importance_df = analyze_model_importance(model, model_name, X_train, y_train, X_test, feature_names)
        all_importances[model_name] = importance_df
        model_names.append(model_name)
    
    # Create combined analysis
    print(f"\nCreating combined feature importance analysis...")
    create_combined_importance_plot(all_importances, model_names)
    
    # Save results to CSV
    for model_name, importance_df in all_importances.items():
        if importance_df is not None:
            importance_df.to_csv(f'feature_importance_{model_name.lower()}.csv', index=False)
            print(f"Saved feature importance for {model_name}")
    
    print(f"\nFeature importance analysis complete!")
    print(f"Generated files:")
    print(f"   - feature_importance_*.png (individual plots)")
    print(f"   - shap_*.png (SHAP plots)")
    print(f"   - combined_feature_importance.png (comparison)")
    print(f"   - feature_importance_*.csv (data files)")

if __name__ == "__main__":
    main()
