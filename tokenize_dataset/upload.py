import click
from huggingface_hub import HfApi, create_repo, upload_file
import os
import glob
from tqdm import tqdm


def upload_to_huggingface(input_dir, repo_id, token=None):
    # Initialize Hugging Face API
    api = HfApi()

    # Create or get repository
    try:
        create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)
    except Exception as e:
        click.echo(f"Note: Repository already exists or error occurred: {e}")

    # Prepare files to upload
    files_to_upload = []

    # Look for safetensors and metadata files
    for pattern in ["*.safetensors", "*_metadata.json"]:
        files = glob.glob(os.path.join(input_dir, pattern))
        files_to_upload.extend(files)

    # Upload each file
    for file_path in tqdm(files_to_upload, desc="Uploading files"):
        file_name = os.path.basename(file_path)
        try:
            click.echo(f"\nUploading {file_name}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            click.echo(f"Successfully uploaded {file_name}")
        except Exception as e:
            click.echo(f"Error uploading {file_name}: {e}")

    click.echo(f"\nAll files uploaded to {repo_id}")
    click.echo(f"View at: https://huggingface.co/datasets/{repo_id}")


@click.command()
@click.option(
    "--input-dir",
    type=str,
    default=".",
    help="Directory containing safetensors and metadata files",
)
@click.option(
    "--repo-id",
    type=str,
    default="fal/cosmos-imagenet",
    help="Hugging Face repository ID",
)
@click.option("--token", type=str, help="Hugging Face API token")
def main(input_dir, repo_id, token):
    """Upload processed files to Hugging Face"""
    upload_to_huggingface(input_dir, repo_id, token)


if __name__ == "__main__":
    main()
