from huggingface_hub import snapshot_download

print("Downloading ReplicaCAD...")
snapshot_download(
    repo_id="ai-habitat/ReplicaCAD_dataset",
    repo_type="dataset",
    local_dir="./replica_cad"
)
print("Done!")
