from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_sample_pdf(file_path):
    c = canvas.Canvas(file_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "SmartDoc AI Test Document")
    c.drawString(100, 730, "Day la tai lieu mau de kiem tra he thong RAG.")
    c.drawString(100, 710, "SmartDoc AI su dung Ollama va LangChain de xu ly van ban.")
    c.drawString(100, 690, "Cau hoi test: He thong nay ten la gi?")
    c.drawString(100, 670, "Tra loi: He thong ten la SmartDoc AI.")
    c.save()

if __name__ == "__main__":
    sample_path = os.path.join("data", "samples", "sample_vi.pdf")
    create_sample_pdf(sample_path)
    print(f"Sample PDF created at {sample_path}")
