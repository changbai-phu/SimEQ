from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent
print("base", BASE_DIR)
# Define path to your document
DISASTER_REPORT_PATH = BASE_DIR / "data" 
# print("data", DISASTER_REPORT_PATH)
document_eval_file = DISASTER_REPORT_PATH / "gemini.pdf"
# print("eval", document_eval_file)
document_sim_file = DISASTER_REPORT_PATH / "Myanmar.pdf"
document_sitrep_file = DISASTER_REPORT_PATH / "Myanmar_SitRep.pdf"
document_guide_file = DISASTER_REPORT_PATH / "EQ_Guidelines.pdf"

