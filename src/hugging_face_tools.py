from pathlib import Path
from datetime import datetime


class HuggingFaceUploader:
    """Класс для загрузки моделей на Hugging Face"""

    def __init__(self, repo_id: str, token: str = None):
        """
        Args:
            repo_id: "username/repo-name" (например "RainbollD/car_dent_segment")
            token: HF токен (или None если уже залогинены через CLI)
        """
        self.repo_id = repo_id
        self.token = token
        self.api = None

    def _get_api(self):
        """Ленивая инициализация API"""
        if self.api is None:
            from huggingface_hub import HfApi
            self.api = HfApi(token=self.token)
        return self.api

    def upload_model(self,
                     model_path: str,
                     commit_message: str = None,
                     create_tag: bool = False,
                     tag_name: str = None,
                     private: bool = False) -> str:
        """
        Загрузка модели на Hugging Face

        Args:
            model_path: путь к папке с моделью
            commit_message: сообщение коммита
            create_tag: создать тег версии
            tag_name: имя тега (например "v1.0.0")
            private: приватный репозиторий

        Returns:
            URL репозитория
        """
        from huggingface_hub import create_repo

        api = self._get_api()
        model_path = Path(model_path)

        # 1. Проверка пути
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # 2. Создание/проверка репозитория
        print(f"🔍 Checking repository {self.repo_id}...")
        try:
            create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            print(f"✅ Repository {self.repo_id} ready")
        except Exception as e:
            print(f"⚠️ Repository might already exist: {e}")

        # 3. Сообщение коммита
        if commit_message is None:
            commit_message = f"Upload model - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # 4. Загрузка файлов
        print(f"📤 Uploading model from {model_path}...")
        try:
            result = api.upload_folder(
                folder_path=str(model_path),
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=["*.git*", "*/.*", "__pycache__", "*.pyc", "*.pyo"]
            )
            print(f"✅ Upload complete!")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise

        # 5. Создание тега (опционально)
        if create_tag and tag_name:
            print(f"🏷️ Creating tag {tag_name}...")
            try:
                api.create_tag(
                    repo_id=self.repo_id,
                    repo_type="model",
                    tag=tag_name,
                    message=f"Release {tag_name}"
                )
                print(f"✅ Tag {tag_name} created!")
            except Exception as e:
                print(f"⚠️ Tag creation failed: {e}")

        # 6. Возврат URL
        url = f"https://huggingface.co/{self.repo_id}"
        print(f"🔗 Model page: {url}")
        if tag_name:
            print(f"🔗 Version: {url}/tree/{tag_name}")

        return url

    def upload_file(self, file_path: str, path_in_repo: str = None):
        """Загрузка отдельного файла"""
        api = self._get_api()
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo or Path(file_path).name,
            repo_id=self.repo_id,
            repo_type="model"
        )


def push_to_huggingface(model_path: str,
                        repo_id: str,
                        token: str = None,
                        tag: str = None,
                        private: bool = False):
    """
    Простая функция для быстрой загрузки модели

    Args:
        model_path: путь к папке с моделью
        repo_id: "username/repo-name"
        token: HF токен (опционально)
        tag: версия модели (например "v1.0.0")
        private: приватный репозиторий
    """
    uploader = HuggingFaceUploader(repo_id=repo_id, token=token)

    commit_msg = f"Model update - {datetime.now().strftime('%Y-%m-%d')}"
    if tag:
        commit_msg += f" [{tag}]"

    uploader.upload_model(
        model_path=model_path,
        commit_message=commit_msg,
        create_tag=(tag is not None),
        tag_name=tag,
        private=private
    )
