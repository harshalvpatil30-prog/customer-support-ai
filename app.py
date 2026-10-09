import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_DIR = "./model"  # relative path — model/ folder must sit next to this script

SYSTEM_PROMPT = (
    "You are a professional, empathetic customer support agent for a bank. "
    "Respond helpfully and concisely to the customer's message, using the "
    "conversation history for context."
)

print("Loading model, please wait...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()
print("Model loaded.")


def generate_response(customer_message, context):
    context = (context or "").strip()
    if context:
        user_content = f"{context}\nClient: {customer_message}\n\nRespond as the Agent to the most recent Client message."
    else:
        user_content = f"Client: {customer_message}\n\nRespond as the Agent to the most recent Client message."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


with gr.Blocks(title="Customer Support AI") as demo:
    gr.Markdown("## 💬 Customer Support AI")
    customer_message = gr.Textbox(label="Customer Message", placeholder="My card hasn't arrived yet...")
    context = gr.Textbox(
        label="Previous Conversation / Context (optional)",
        placeholder="Agent: Hi, how can I help?\nClient: ...",
        lines=4,
    )
    generate_btn = gr.Button("Generate Response")
    response = gr.Textbox(label="AI Agent Response", lines=4, interactive=False)

    generate_btn.click(fn=generate_response, inputs=[customer_message, context], outputs=response)

if __name__ == "__main__":
    demo.launch()
