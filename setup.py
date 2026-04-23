"""
Утилита для установки и инициализации проекта
"""
import os
import sys
import subprocess
from pathlib import Path


class ProjectSetup:
    """Установка и инициализация проекта"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
    
    def check_python_version(self):
        """Проверить версию Python"""
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print(f"✓ Python {sys.version.split()[0]} OK")
        return True
    
    def check_env_file(self):
        """Проверить наличие .env"""
        env_file = self.project_root / ".env"
        if not env_file.exists():
            print("❌ .env file not found")
            print("Creating .env from template...")
            
            env_template = """# API Keys
API_KEY_SPORTS=your_api_key_here
API_KEY_FOOTBALL_DATA=your_api_key_here

# Settings
LOG_LEVEL=INFO
SCHEDULER_ENABLED=True
"""
            env_file.write_text(env_template)
            print("✓ Created .env (edit with your API keys)")
            return False
        
        # Проверить наличие ключей
        env_content = env_file.read_text()
        has_sports_key = "your_api_key_here" not in env_content.split("API_KEY_SPORTS")[1].split("\n")[0]
        
        if has_sports_key:
            print("✓ .env found with API keys")
            return True
        else:
            print("⚠ .env found but API keys not configured")
            print("  Edit .env and add your API keys from:")
            print("  - api-sports.io")
            print("  - football-data.org")
            return False
    
    def install_dependencies(self):
        """Установить зависимости"""
        print("Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_file)
            ])
            print("✓ Dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def create_directories(self):
        """Создать необходимые директории"""
        dirs = [
            self.project_root / "data_cache",
            self.project_root / "models",
            self.project_root / "database",
            self.project_root / "logs",
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(exist_ok=True)
            print(f"✓ Created {dir_path.name}/")
    
    def run_setup(self):
        """Запустить полную установку"""
        print("\n🚀 Football Predictions System Setup\n")
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directories),
            ("Installing dependencies", self.install_dependencies),
            ("Checking .env file", self.check_env_file),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            try:
                result = step_func()
                if not result and step_name == "Checking .env file":
                    print("⚠ Continuing without API keys (will need them to fetch data)")
                elif not result:
                    print(f"❌ Setup failed at: {step_name}")
                    return False
            except Exception as e:
                print(f"❌ Error during {step_name}: {e}")
                return False
        
        print("\n" + "="*50)
        print("✅ Setup completed successfully!")
        print("="*50)
        print("\nNext steps:")
        print("1. Edit .env with your API keys")
        print("2. Run: streamlit run main.py")
        print("\nAPI keys from:")
        print("  - https://dashboard.api-sports.io/register")
        print("  - https://www.football-data.org/client/register")
        print("")
        
        return True


def main():
    """Главная функция"""
    setup = ProjectSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
