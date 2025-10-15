import pandas as pd, joblib, argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from preprocessing import build_preprocessor
from model import build_models
import mlflow
import os

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics"""
    # Get predictions
    print("In Evaluate Model function")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }


def main(data_path, output, experiment_name):
    df = pd.read_csv(data_path,low_memory=False)
    # Convert all object columns to string and fill missing with 'Unknown'
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).fillna('Unknown')

    
    mixed_cols = [1,7,8,16,17,18,19,20,35]  
    for c in mixed_cols:
        df.iloc[:, c] = df.iloc[:, c].astype(str).fillna('Unknown')

    target = 'Default'
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    num = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat = X.select_dtypes(include=['object']).columns.tolist()
    pre = build_preprocessor(num, cat, [])

    # Get all models
    models = build_models(pre)
    
    # Train and evaluate all models
    results = []
    best_model = None
    best_f1 = 0

    print("Training and evaluating models...")
    print("=" * 80)
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        
        # Print metrics for this model
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("-" * 40)

        # Track best model
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model = model
    
    # Print summary table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model']:<20} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} {result['f1_score']:<10.4f} {result['roc_auc']:<10.4f}")
    
    # Save best model
    os.makedirs(output, exist_ok=True)
    joblib.dump(best_model, f'{output}/best_pipeline.joblib')
    print(f"\nBest model ({best_model.named_steps['model'].__class__.__name__}) saved to {output}/best_pipeline.joblib")
    
    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output}/model_comparison.csv', index=False)
    print(f"Model comparison results saved to {output}/model_comparison.csv")



   

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path',default="data/dataset.csv")
    p.add_argument('--output', default='out/')
    p.add_argument('--experiment-name', default='loan-default')
    a = p.parse_args()
    main(a.data_path, a.output, a.experiment_name)