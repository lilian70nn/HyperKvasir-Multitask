import gradio as gr
import pandas as pd

from src.vlm.model import EndoscopyAssistantModel


print("Loading Endoscopy Assistant...")
assistant = EndoscopyAssistantModel()
print("Endoscopy Assistant ready.")


def build_prediction_table(result):
    evidence = result.get("evidence")

    if evidence is None:
        return pd.DataFrame(columns=["Field", "Value"])

    classification = evidence.get("classification", {})
    segmentation = evidence.get("segmentation")

    rows = [
        {"Field": "Predicted class", "Value": str(evidence.get("label", ""))},
        {"Field": "Confidence", "Value": f"{float(evidence.get('confidence', 0.0)) * 100:.2f}%"},
        {"Field": "Classification route", "Value": str(classification.get("route", ""))},
        {"Field": "Segmentation", "Value": "Available" if segmentation is not None else "Not generated"},
    ]

    if classification.get("specialist_confidence") is not None:
        rows.append({"Field": "Specialist confidence", "Value": f"{float(classification['specialist_confidence']) * 100:.2f}%"})

    return pd.DataFrame(rows)


def run_inference(text, image):
    text = text.strip() if text else None

    if text is None and image is None:
        return "Please enter a question, upload an image, or provide both.", pd.DataFrame(columns=["Field", "Value"]), None, "No input was provided."

    try:
        result = assistant.predict(text=text, image=image)
    except Exception as error:
        return f"An error occurred during inference:\n\n{error}", pd.DataFrame(columns=["Field", "Value"]), None, "Inference failed."

    answer = result["answer"]
    details = build_prediction_table(result)
    annotated_image = result.get("annotated_image")

    if result.get("used_endoscopy_vision"):
        status = "Specialized Endoscopy Vision Module was used."
    else:
        status = "Answered by the general-purpose VLM without specialist analysis."

    return answer, details, annotated_image, status


def clear_interface():
    return None, "", "", pd.DataFrame(columns=["Field", "Value"]), None, ""


CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
}
#title {
    text-align: center;
    margin-bottom: 4px;
}
#subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 24px;
}
#answer-box {
    min-height: 170px;
}
#status-box {
    font-size: 0.9rem;
    color: #666;
}
"""


with gr.Blocks(css=CSS, title="Endoscopy AI Assistant") as demo:
    gr.Markdown("# Endoscopy AI Assistant", elem_id="title")
    gr.Markdown("Ask a question, upload an endoscopy image, or provide both. Specialized classification and polyp segmentation are used automatically when relevant.", elem_id="subtitle")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", height=360)
            text_input = gr.Textbox(label="Question", placeholder="For example: What do you see in this endoscopy image?", lines=4)

            with gr.Row():
                analyze_button = gr.Button("Analyze", variant="primary")
                clear_button = gr.Button("Clear")

        with gr.Column(scale=1):
            answer_output = gr.Markdown(label="Answer", elem_id="answer-box")
            status_output = gr.Markdown(elem_id="status-box")

    gr.Markdown("### Prediction Details")
    prediction_table = gr.Dataframe(
        headers=["Field", "Value"],
        datatype=["str", "str"],
        interactive=False,
        wrap=True,
    )

    gr.Markdown("### Highlighted Region")
    annotated_output = gr.Image(type="pil", label="Predicted Region", height=420)

    analyze_button.click(
        fn=run_inference,
        inputs=[text_input, image_input],
        outputs=[answer_output, prediction_table, annotated_output, status_output],
    )

    text_input.submit(
        fn=run_inference,
        inputs=[text_input, image_input],
        outputs=[answer_output, prediction_table, annotated_output, status_output],
    )

    clear_button.click(
        fn=clear_interface,
        inputs=[],
        outputs=[image_input, text_input, answer_output, prediction_table, annotated_output, status_output],
    )


if __name__ == "__main__":
    demo.launch(share=True)