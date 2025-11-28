import gradio as gr
import json
import subprocess
import os
from PIL import Image
from pathlib import Path

def analyze_battle(image):
    """Analyze a Pokemon battle and return the best move"""
    
    if image is None:
        return "⚠️ Please upload an image first!", ""
    
    temp_path = "/tmp/battle_temp.png"
    image.save(temp_path)
    
    try:
        # Get the directory where app.py is located
        app_dir = Path(__file__).parent.absolute()
        
        result = subprocess.run(
            ["pokeai", "run", "--image", temp_path, "--config", "configs/default.yml"],
            capture_output=True,
            text=True,
            cwd=str(app_dir)
        )
        
        output = result.stdout
        recognition_json = None
        decision_json = None
        sections = output.split('===')
        
        for i, section in enumerate(sections):
            if 'RECOGNITION' in section:
                json_text = sections[i+1].strip() if i+1 < len(sections) else ""
                end_idx = json_text.find('\n\n')
                if end_idx > 0:
                    json_text = json_text[:end_idx]
                try:
                    recognition_json = json.loads(json_text)
                except:
                    pass
            elif 'DECISION' in section:
                json_text = sections[i+1].strip() if i+1 < len(sections) else ""
                end_idx = json_text.find('\n\n')
                if end_idx > 0:
                    json_text = json_text[:end_idx]
                try:
                    decision_json = json.loads(json_text)
                except:
                    pass
        
        if recognition_json and decision_json:
            our_pokemon = recognition_json.get('our_pokemon', 'N/A')
            opponent_pokemon = recognition_json.get('opponent_pokemon', 'N/A')
            our_hp = recognition_json.get('our_hp_percent', 0)
            moves = recognition_json.get('moves_ocr', [])
            
            reasoning_data = decision_json.get('reasoning', {})
            moves_ranking = reasoning_data.get('available_moves_ranking_head', [])
            
            best_move = moves_ranking[0]['move_name'] if len(moves_ranking) > 0 else 'N/A'
            second_move = moves_ranking[1]['move_name'] if len(moves_ranking) > 1 else None
            
            moves_html = ""
            for i, move in enumerate(moves, 1):
                if move == best_move:
                    moves_html += f"<div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 12px 15px; margin: 8px 0; border-radius: 8px; color: white; font-weight: bold; box-shadow: 0 2px 8px rgba(17,153,142,0.3);'>{i}. ⭐ {move}</div>"
                elif move == second_move:
                    moves_html += f"<div style='background: #fff3e0; padding: 12px 15px; margin: 8px 0; border-radius: 8px; border: 2px solid #ff9800; font-weight: 600; color: #e65100;'>{i}. {move}</div>"
                else:
                    moves_html += f"<div style='background: #f5f5f5; padding: 12px 15px; margin: 8px 0; border-radius: 8px; border: 1px solid #e0e0e0; color: #666;'>{i}. {move}</div>"
            
            if not moves_html:
                moves_html = '<p style="color: #999;">No moves detected</p>'
            
            second_move_html = ""
            if second_move:
                second_move_html = f"""
                <div style='background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); padding: 20px; border-radius: 12px; margin: 15px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.15); color: white; text-align: center;'>
                    <h3 style='margin: 0 0 8px 0; font-size: 16px; opacity: 0.9;'>🥈 Alternative</h3>
                    <p style='font-size: 32px; margin: 8px 0; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{second_move}</p>
                </div>
                """
            
            result_html = f"""
            <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 650px; margin: 0 auto; padding: 20px;'>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px;'>
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: white;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>👤 Your Pokemon</div>
                        <div style='font-size: 32px; font-weight: bold; margin: 8px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{our_pokemon}</div>
                        <div style='background: rgba(255,255,255,0.2); padding: 8px 12px; border-radius: 6px; display: inline-block; margin-top: 8px;'>
                            <span style='font-size: 18px;'>❤️ {our_hp:.0f}%</span>
                        </div>
                    </div>
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: white;'>
                        <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>⚔️ Opponent</div>
                        <div style='font-size: 32px; font-weight: bold; margin: 8px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{opponent_pokemon}</div>
                    </div>
                </div>
                <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 30px; border-radius: 15px; margin: 25px 0; box-shadow: 0 6px 25px rgba(17,153,142,0.3); color: white; text-align: center;'>
                    <h2 style='margin: 0 0 12px 0; font-size: 18px; opacity: 0.95; letter-spacing: 1px;'>🎯 RECOMMENDED MOVE</h2>
                    <p style='font-size: 56px; margin: 15px 0; font-weight: 900; text-shadow: 3px 3px 6px rgba(0,0,0,0.3); letter-spacing: 3px;'>{best_move}</p>
                </div>
                {second_move_html}
                <div style='background: white; padding: 20px; border-radius: 12px; margin: 20px 0; box-shadow: 0 3px 12px rgba(0,0,0,0.08); border: 2px solid #f0f0f0;'>
                    <h3 style='margin: 0 0 15px 0; color: #333; font-size: 18px; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px;'>📋 All Moves</h3>
                    {moves_html}
                </div>
            </div>
            """
            return result_html, output
        else:
            return "❌ Error parsing JSON output", output
    except Exception as e:
        return f"<div style='background: #ffcdd2; padding: 20px; border-radius: 12px;'><h3>❌ Error</h3><p>{str(e)}</p></div>", str(e)

with gr.Blocks(title="Pokemon Battle Analyzer") as demo:
    gr.Markdown("# 🎮 Pokemon Battle Analyzer")
    gr.Markdown("### Upload a battle screenshot to get move recommendations!")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📸 Battle Screenshot")
            analyze_btn = gr.Button("🔍 Analyze Battle", variant="primary", size="lg")
        with gr.Column(scale=1):
            result_output = gr.HTML(label="📊 Analysis")
    with gr.Accordion("🔧 Technical Output", open=False):
        debug_output = gr.Textbox(label="Debug Details", lines=15)
    analyze_btn.click(fn=analyze_battle, inputs=image_input, outputs=[result_output, debug_output])
    gr.Markdown("---")
    gr.Markdown("💡 **Tip**: Use clear screenshots for best results!")

demo.launch(server_port=7861)
EOF
