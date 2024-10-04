from moviepy.editor import VideoFileClip
import whisper

def video_to_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)



def audio_to_text(audio_path, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"]

def video_to_text(video_path, audio_path, model_name="base"):
    # Convert video to audio
    video_to_audio(video_path, audio_path)

    # Convert audio to text
    text = audio_to_text(audio_path, model_name)

    return text

