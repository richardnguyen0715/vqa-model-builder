import matplotlib.pyplot as plt
import seaborn as sns

from src.middleware.logger import data_process_logger as logging
from utils.path_management import ROOT_DIR

sample_image_path = ROOT_DIR / 'data' / 'raw' / 'images' / '000000581569.jpg'
question_example = "Mối quan hệ giữa người trượt ván và đám đông là gì?"
answer_example = "['Người trượt ván biểu diễn cho đám đông', 'Đám đông đang quan sát vận động viên', 'Khán giả đang xem người trượt ván', 'Người trượt ván đang biểu diễn trước khán giả', 'Đám đông là khán giả của người trượt ván']"


def show_sample(image, question, answer, is_save: bool = False, save_path: str = ROOT_DIR / "tests" / "sample_visualization.png"):
    """Display a sample image with its question and answer.

    Args:
        image: The image to display (as a NumPy array).
        question: The question related to the image.
        answer: The answer to the question.
    """
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Q: {question}\nA: {answer}")
    plt.show(block=True)
    if is_save:
        plt.savefig(save_path)
        logging.info(f"Sample visualization saved to {save_path}")

    
if __name__ == "__main__":
    # Load sample image
    import cv2
    image = cv2.imread(str(sample_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logging.info(f"Loaded image from {sample_image_path}")
    # Show sample with question and answer
    show_sample(image, question_example, answer_example, is_save=True)