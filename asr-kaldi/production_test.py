from gradio_client import Client, handle_file

client = Client("https://sapolita-kaldi.ithuan.tw/")
result = client.predict(
    model_id="formosan-kaldi-250514",
    dialect_id="formosan_ami",
    audio_data=handle_file(
        'https://klokah.tw/extension/sp_senior/sound/2/2sentence/3_1.mp3'
    ),
    api_name="/predict"
)
print(result)
