"""风格分类器"""

class StyleClassifier:
    def classify(self, features: dict) -> dict:
        """分类风格"""
        scores = {}
        
        if 150 < features["f0_mean"] < 300:
            scores["pop_ballad"] = 0.89
        if features["f0_range"][1] - features["f0_range"][0] < 200:
            scores["folk_acoustic"] = 0.76
        
        return scores
