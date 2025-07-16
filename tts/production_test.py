from gradio_client import Client, handle_file

client = Client("https://hnang-kari-ai-asi-sluhay.ithuan.tw/")
result = client.predict(
    model_drop_down="all-formosan-v2-step-843031",
    language="阿美_秀姑巒",
    ref_audio_input=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
    ref_text_input="Hello",
    gen_text_input="Hello",
    remove_silence=False,
    cross_fade_duration_slider=0.15,
    nfe_slider=32,
    speed_slider=1,
    api_name="/basic_tts"
)
print(result)
