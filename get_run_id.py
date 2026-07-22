# get_run_id.py
import mlflow

# Get all experiments
client = mlflow.MlflowClient()
experiments = client.search_experiments()

print("=" * 60)
print("📊 Available Experiments:")
print("=" * 60)

for exp in experiments:
    print(f"\n🔹 ID: {exp.experiment_id}")
    print(f"   Name: {exp.name}")
    print(f"   Lifecycle Stage: {exp.lifecycle_stage}")
    
    # Get runs in this experiment
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=5
    )
    
    if runs:
        print(f"   📋 Runs ({len(runs)} total):")
        for i, run in enumerate(runs, 1):
            print(f"      {i}. Run ID: {run.info.run_id}")
            print(f"         Status: {run.info.status}")
            
            # Get run name
            run_name = run.data.tags.get('mlflow.runName', 'Unknown')
            print(f"         Name: {run_name}")
            
            # Get model type
            model_type = run.data.params.get('model_type', 'Unknown')
            print(f"         Model: {model_type}")
            print()
    else:
        print("   ⚠️  No runs found in this experiment")

print("=" * 60)
print("\n💡 Tip: Copy the Run ID above to use with:")
print("   mlflow models serve -m runs:/<run_id>/model -p 5001")
print("=" * 60)