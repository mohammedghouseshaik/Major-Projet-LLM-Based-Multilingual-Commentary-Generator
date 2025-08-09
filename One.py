import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define the folder where JSON files are stored
DATA_FOLDER = "F:\\Major Project\\Least json files"

# Define the model (can replace with a fine-tuned model)
MODEL_NAME = "gpt2"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Load all JSON files
def load_json_files(folder):
    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    all_matches = []
    
    for file in tqdm(json_files, desc="Loading JSON Files"):
        file_path = os.path.join(folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            match_data = json.load(f)
            all_matches.append(match_data)
    
    return all_matches

# Extract ball-by-ball data
def extract_match_data(match_data):
    commentary_data = []
    for inning in match_data.get("innings", []):
        team = inning.get("team", "Unknown Team")
        for over in inning.get("overs", []):
            over_num = over.get("over", 0)
            for ball in over.get("deliveries", []):
                ball_info = {
                    "team": team,
                    "over": over_num,
                    "batter": ball.get("batter", "Unknown Batter"),
                    "bowler": ball.get("bowler", "Unknown Bowler"),
                    "runs": ball["runs"].get("batter", 0),
                    "extras": ball["runs"].get("extras", 0),
                    "total_runs": ball["runs"].get("total", 0),
                    "wicket": ball.get("wickets", None)
                }
                commentary_data.append(ball_info)
    
    return commentary_data

# Define commentary styles
COMMENTARY_STYLES = [
    "Traditional Ball-by-Ball",
    "Analytical Perspective",
    "Excited, Energetic Commentary",
    "Dramatic Storytelling",
    "Historical Perspective",
    "Humorous Commentary"
]

def generate_commentary(ball_data):
    commentary_outputs = {}

    # Check if wicket exists and extract details safely
    if ball_data.get("wicket") and isinstance(ball_data["wicket"], list) and len(ball_data["wicket"]) > 0:
        wicket_info = ball_data["wicket"][0]  # Extract first wicket entry
        kind = wicket_info.get("kind", "unknown dismissal")
        fielders = ", ".join(f["name"] for f in wicket_info.get("fielders", []))
        wicket_text = f"OUT! {kind} by {fielders}" if fielders else f"OUT! {kind}"
    else:
        wicket_text = "No wicket."

    input_text = (
        f"{ball_data['batter']} faces {ball_data['bowler']}. "
        f"Runs scored: {ball_data['total_runs']}. {wicket_text}"
    )

    for style in COMMENTARY_STYLES:
        prompt = f"Generate a {style} cricket commentary for: {input_text}"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=75)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=75,  # Adjust length for concise responses
                do_sample=True, 
                temperature=0.9,  # Increases variation
                top_p=0.9,        # Nucleus sampling
                top_k=50          # Helps filter out improbable words
            )

        commentary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing cleanup if needed
        commentary = commentary.replace(prompt, "").strip()
        
        commentary_outputs[style] = commentary
    
    return commentary_outputs


# Main function to run commentary generation
def main():
    matches = load_json_files(DATA_FOLDER)
    
    all_commentaries = []
    
    for match in tqdm(matches, desc="Processing Matches"):
        match_commentary = []
        match_data = extract_match_data(match)
        
        for ball in match_data:
            commentary = generate_commentary(ball)
            match_commentary.append({
                "over": ball["over"],
                "batter": ball["batter"],
                "bowler": ball["bowler"],
                "runs": ball["total_runs"],
                "wicket": ball["wicket"],
                "commentary": commentary
            })
        
        all_commentaries.append(match_commentary)
    
    # Save output to a JSON file
    with open("generated_commentary.json", "w", encoding="utf-8") as f:
        json.dump(all_commentaries, f, indent=4)

    print("Commentary generation complete! Output saved to generated_commentary.json")

if __name__ == "__main__":
    main()
