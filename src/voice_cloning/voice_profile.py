"""声音档案管理"""
import json

class VoiceProfile:
    def save(self, profile: dict, user_id: str):
        """保存声音档案"""
        path = f"models/user_voices/{user_id}_profile.json"
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
    
    def load(self, user_id: str) -> dict:
        """加载声音档案"""
        path = f"models/user_voices/{user_id}_profile.json"
        with open(path, "r") as f:
            return json.load(f)
