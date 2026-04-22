from datetime import datetime
from pathlib import Path


class HuggingFaceUploader:
    def __init__(self, repo_id: str, token: str = None):
        self.repo_id = repo_id
        self.token = token
        self._api = None

    def _get_api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    def upload_model(
        self,
        model_path: str,
        commit_message: str = None,
        create_tag: bool = False,
        tag_name: str = None,
        private: bool = False,
    ) -> str:
        from huggingface_hub import create_repo

        api = self._get_api()
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        create_repo(repo_id=self.repo_id, repo_type="model", private=private, exist_ok=True)

        if commit_message is None:
            commit_message = f"Upload model - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        api.upload_folder(
            folder_path=str(model_path),
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.git*", "*/.*", "__pycache__", "*.pyc", "*.pyo"],
        )

        if create_tag and tag_name:
            try:
                api.create_tag(
                    repo_id=self.repo_id,
                    repo_type="model",
                    tag=tag_name,
                    message=f"Release {tag_name}",
                )
            except Exception as exc:
                print(f"Tag creation failed: {exc}")

        url = f"https://huggingface.co/{self.repo_id}"
        return url

    def upload_file(self, file_path: str, path_in_repo: str = None) -> None:
        api = self._get_api()
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo or Path(file_path).name,
            repo_id=self.repo_id,
            repo_type="model",
        )


def push_to_huggingface(
    model_path: str,
    repo_id: str,
    token: str = None,
    tag: str = None,
    private: bool = False,
) -> str:
    uploader = HuggingFaceUploader(repo_id=repo_id, token=token)
    commit_msg = f"Model update - {datetime.now().strftime('%Y-%m-%d')}"
    if tag:
        commit_msg += f" [{tag}]"
    return uploader.upload_model(
        model_path=model_path,
        commit_message=commit_msg,
        create_tag=(tag is not None),
        tag_name=tag,
        private=private,
    )
