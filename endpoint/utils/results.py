from utils.logger import logger

def format_results(results):
    """
    Format result as a key value paire where :
        - k -> sentiment
        - v -> prob
    """
    predictions = {}
    names: dict = results[0].names
    probs: list = results[0].probs.data.tolist()
    for idx, sentiment in names.items():
        predictions[sentiment] = probs[idx]
    return predictions