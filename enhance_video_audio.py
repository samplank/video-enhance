import os
import math
import time
import requests
import shutil
import json
import replicate
import numpy as np
import gc
import cv2
import asyncio
import aiohttp
import re
from concurrent.futures import ThreadPoolExecutor
import shlex
import itertools
from moviepy.video.io.VideoFileClip import VideoFileClip
from dotenv import load_dotenv
import argparse
from utils import *

load_dotenv()

# Set env variables in terminal with export VARIABLE_NAME=VALUE
DOLBY_APP_KEY = os.environ["DOLBY_APP_KEY"]
DOLBY_APP_SECRET = os.environ["DOLBY_APP_SECRET"]
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_ENHANCE_MODEL = os.environ.get('REPLICATE_ENHANCE_MODEL')
BUCKET = os.environ.get("BUCKET")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")


payload = { 'grant_type': 'client_credentials', 'expires_in': 1800}

response = requests.post('https://api.dolby.io/v1/auth/token', data=payload, auth=requests.auth.HTTPBasicAuth(DOLBY_APP_KEY,DOLBY_APP_SECRET))
body = json.loads(response.content)
access_token = body['access_token']


def get_video_duration(chunk_filename):
    cmd = f"ffprobe -v quiet -print_format json -show_format -show_streams -i {chunk_filename}"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    video_info = json.loads(output)
    print(video_info)
    duration = float(video_info["format"]["duration"])
    return duration

async def run_ffmpeg_command(cmd, retries=3, delay=5):
    cmd_args = shlex.split(cmd)

    for attempt in range(retries + 1):
        process = await asyncio.create_subprocess_exec(
            *cmd_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            print(f"Command executed successfully: {stdout.decode()}")
            break
        else:
            if attempt < retries:
                print(f"An error occurred: {stderr.decode()}")
                print(f"Retrying in {delay} seconds... (attempt {attempt + 1} of {retries})")
                await asyncio.sleep(delay)
            else:
                print(f"An error occurred after {retries} retries: {stderr.decode()}")
                break

async def fetch_upgraded_frame(session, src, filename, queue, logger):
    logger.info(f"Starting to fetch frame: {filename}")
    timeout = aiohttp.ClientTimeout(total=120)  # Adjust the value as needed
    async with session.get(src, timeout=timeout) as response:
        logger.info(f"Response received for frame: {filename}")
        buffer = []
        async for chunk in response.content.iter_chunked(4096):
            logger.info(f"Processing chunk for frame: {filename}")
            buffer.append(chunk)

        logger.info(f"Finished processing chunks for frame: {filename}")
        image_data = np.fromiter(itertools.chain(*buffer), dtype=np.uint8)
        del buffer
        upgraded_frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        del image_data
        await queue.put((filename, upgraded_frame))
        del upgraded_frame

async def process_upgraded_frame(semaphore, i, video_reader, output_folder, frame_queue, logger, max_retries=3):
    async with semaphore:
        print(f"process_upgraded_frame: Start, frame {i}")
        frame_data = None
        for attempt in range(max_retries):
            try:
                # Set the video reader to the correct frame
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame_data = video_reader.read()

                # Save the frame_data to a temporary file
                temp_filename = os.path.join(output_folder, f"frame_{i}.png")
                cv2.imwrite(temp_filename, frame_data)

                # Run start_upgrade on the saved frame
                # image, f = start_upgrade(temp_filename)
                image, f = await start_upgrade_async(temp_filename)

                # If start_upgrade was successful, run finish_upgrade_frame
                if image is not None and f is not None:
                    upgraded_frame, src = await finish_upgrade_frame(image, f)

                    if upgraded_frame is not None:
                        # Put the upgraded frame and its filename into the queue
                        upgraded_filename = f'upgraded_frame_{i}.png'
                        await frame_queue.put((upgraded_filename, upgraded_frame))
                        print(f"Frame {i} added to queue")
                        os.remove(temp_filename)
                        del frame_data
                        break
                # Remove the temporary file
                os.remove(temp_filename)
                del frame_data
            except Exception as e:
                print('process_upgraded_frame exception on:' + str(i))
                print(e)
                if attempt == max_retries - 1:
                    print('hit max number of retries on ' + str(i))
                    # Put the original frame and its filename into the queue if frame_data is not None
                    if frame_data is not None:
                        original_filename = f'original_frame_{i}.png'
                        await frame_queue.put((original_filename, frame_data))
                        print(f"Original Frame {i} added to queue")
                        os.remove(temp_filename)
                        del frame_data
                    else:
                        print("frame_data is none " + str(i))
                    break
                else:
                    await asyncio.sleep(1)

        print(f"process_upgraded_frame: End, frame {i}")

async def write_frames_to_output(queue, out, total_frames, logger):
    print('start write_frames_to_output')
    written_frames = 0
    frame_buffer = {}

    while written_frames < total_frames:
        filename, upgraded_frame = await queue.get()
        print(f'Got frame from queue, queue size: {queue.qsize()}, frame buffer size: {len(frame_buffer)}')
        try:
            frame_index = int(filename.split('_')[2].split('.')[0])
            print("write_frames_to_output index: " + str(frame_index))

        except ValueError:
            print(f"Error: Invalid filename format: {filename}")
            continue

        frame_buffer[frame_index] = upgraded_frame

        while written_frames in frame_buffer:
            # Check if the frame is None or empty
            if np.all(upgraded_frame == 0):
                print('Error: Frame is None or empty')
            else:
                print(f'Writing frame {written_frames} to output, frame size: {len(upgraded_frame)}')

            out.write(frame_buffer[written_frames])
            logger.info(f"Frame written to output: {filename}")
            del frame_buffer[written_frames]
            written_frames += 1
            print(f'Written_frames count: {written_frames}')

        print(f'End of while loop, frame_buffer size: {len(frame_buffer)}')

    print(f"write_frames_to_output: End")

def process_frames_in_parallel(upgraded_frames, out, logger):
    for i, image in enumerate(upgraded_frames):
        # Add debugging statements
        print(f"Processing frame {i}...")
        print(f"Image type: {type(image)}")

        if isinstance(image, tuple):
            print(f"Image content (tuple): {image}")

        if isinstance(image, np.ndarray):
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")

        # Check if the image is a valid numpy array and has the correct number of channels (3)
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            out.write(image)
            del image
        else:
            logger.error(f"Invalid image at frame {i}. Skipping this frame.")

def stitch_frames_to_video(upgraded_frames, output_video_path, fps, frame_size, logger):
    logger.info('Stitching frames to video...')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    process_frames_in_parallel(upgraded_frames, out, logger)

    out.release()

def enhance_audio(input_filename, output_filename):
    max_retries = 3

    print('enhance_audio input_filename' + input_filename)

    cwd = os.getcwd()
    print('Current working directory:', cwd)
    print(os.listdir(cwd))

    # Set or replace these values
    body = {
      "input" : "https://storage.googleapis.com/writersvoice/" + input_filename,
      "output" : "dlb://out/" + output_filename,
      "content" : {
          "type": "podcast"
      }
    }

    url = "https://api.dolby.com/media/enhance"
    headers = {
        "Authorization": "Bearer {0}".format(access_token),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    for retry in range(max_retries):
        try:
            response = requests.post(url, json=body, headers=headers)
            response.raise_for_status()
            job_id = response.json()['job_id']

            print('this is job_id')
            print(job_id)
            break
        except Exception as e:
            print(f"Retry {retry + 1} failed with exception: {e}")
            if retry == max_retries - 1:
                raise
            time.sleep(2 ** retry)

    status = ''
        
    while status != 'Success':
        for retry in range(max_retries):
            try:
                url = "https://api.dolby.com/media/enhance"
                headers = {
                    "Authorization": "Bearer {0}".format(access_token),
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }

                # TODO: You must replace this value with the job ID returned from the previous step.
                params = {
                  "job_id": job_id
                }

                response = requests.get(url, params=params, headers=headers)
                response.raise_for_status()
                status = response.json()['status']
                print(response.json())
                print(status)
                time.sleep(1)
                break
            except Exception as e:
                print(f"Retry {retry + 1} failed with exception: {e}")
                if retry == max_retries - 1:
                    raise
                time.sleep(2 ** retry)
        
    for retry in range(max_retries):
        try:
            url = "https://api.dolby.com/media/output"
            headers = {
                "Authorization": "Bearer {0}".format(access_token),
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            args = {
                "url": "dlb://out/" + output_filename,
            }

            with requests.get(url, params=args, headers=headers, stream=True) as response:
                response.raise_for_status()
                response.raw.decode_content = True
                print("Downloading from {0} into {1}".format(response.url, output_filename))
                with open(output_filename, "wb") as output_file:
                    shutil.copyfileobj(response.raw, output_file)
            break
        except Exception as e:
            print(f"Retry {retry + 1} failed with exception: {e}")
            if retry == max_retries - 1:
                raise
            time.sleep(2 ** retry)

async def start_upgrade_async(f):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, start_upgrade, f)
    return result

def start_upgrade(f):
    replicate_model = replicate.models.get(REPLICATE_ENHANCE_MODEL)

    if os.path.isfile(f):
        print('is_file')
        print(f)
        try:
            with open(f, "rb") as file:
                image = replicate.predictions.create(
                    version=replicate_model.versions.list()[0],
                    input={
                        "img": file,
                        "version": "v1.2",
                        "scale": 2
                    }
                )
            file.close()
            return (image, f)

        except Exception as e:
            print(f'Error on start_upgrade_image_to_replicate: {str(e)}')
    else:
        print(f"File not found: {f}")

    return (None, None)

async def finish_upgrade_frame(image, filename):
    i = 0
    while (i < 10):
        image.reload()
        print(filename)
        if image.status == 'succeeded':
            print()
            src = image.output
            new_filename = 'upgraded_' + filename.split('/')[1]
            print(new_filename)
            response = requests.get(src)
            
            # Read the image as a numpy array instead of writing it to disk
            image_data = np.frombuffer(response.content, dtype=np.uint8)
            upgraded_frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            print('Memory usage: ' + str(process.memory_info().rss * 0.000001) + ' MB')

            # Free up memory by deleting response and image_data, and running the garbage collector
            del response
            del image_data
            
            return upgraded_frame, src

        i += 1
        await asyncio.sleep(1)

    print(f"Error: Failed to process frame {filename}")
    return None, None

async def upgrade_audio_video(
        input_filename, 
        output_filename, 
        duration_seconds
    ):

    # download_from_gs(BUCKET, input_filename, input_filename)
    has_audio = has_audio_stream(input_filename)

    if has_audio:
        intermediate_filename = 'intermediate_' + input_filename
        audio_only = "audio_only" + output_filename.split('.')[0] + '.aac'
        enhance_audio(input_filename, intermediate_filename)
        await run_ffmpeg_command("""ffmpeg -y -i """ + intermediate_filename + """ -vn -acodec aac """ + audio_only)
        os.remove(input_filename)
    else:
        intermediate_filename = input_filename

    video_reader = cv2.VideoCapture(intermediate_filename)

    # Get the video metadata
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    # Get the height and width from the frame's shape
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Create the frame size tuple
    frame_size = (width * 2, height * 2)

    os.remove(intermediate_filename)

    output_folder = os.path.abspath("frames")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename_pre = 'pre_' + output_filename
    out = cv2.VideoWriter(output_filename_pre, fourcc, fps, frame_size)

    frame_queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(10)  # Adjust the number of concurrent frames being processed as needed

    async def process_frames(semaphore, video_reader, output_folder, frame_queue, logger, max_retries=3):
        tasks = []
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(total_frames):
            task = asyncio.create_task(process_upgraded_frame(semaphore, i, video_reader, output_folder, frame_queue, logger, max_retries))
            tasks.append(task)

        await asyncio.gather(*tasks)

    process_frames_task = process_frames(semaphore, video_reader, output_folder, frame_queue, logger)
    write_frames_task = asyncio.create_task(write_frames_to_output(frame_queue, out, int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)), logger))

    await asyncio.gather(process_frames_task, write_frames_task)

    video_reader.release()

    out.release()

    del out
    gc.collect()
    
    if has_audio:

        upload_to_gs(BUCKET, output_filename_pre, output_filename_pre)

        await run_ffmpeg_command("""ffmpeg -y -i """ + output_filename_pre+ """ -i """ + audio_only + """ -c:v libx264 -c:a aac -map 0:v:0 -map 1:a:0 """ + output_filename)

        os.remove(audio_only)

        upload_to_gs(BUCKET, output_filename, output_filename)

    elif not has_audio:

        run_ffmpeg_command("""ffmpeg -i """ + output_filename_pre + """ -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a aac -b:a 128k """ + output_filename)

        upload_to_gs(BUCKET, output_filename, output_filename)

    os.remove(output_filename_pre)

async def split_video(input_filename, chunk_duration, output_filename, duration_seconds):
    num_chunks = math.ceil(duration_seconds / chunk_duration)
    segment_times = ",".join([str(i * chunk_duration) for i in range(1, num_chunks)])

    command = f"ffmpeg -i {input_filename} -c:v libx264 -crf 23 -f segment -segment_time {chunk_duration} -reset_timestamps 1 chunk_%d_{input_filename}"
    await run_ffmpeg_command(command)

    pattern = re.compile(f"chunk_\d+_{input_filename}")

    chunk_count = 0
    for file in os.listdir(os.getcwd()):
        if pattern.match(file):
            chunk_count += 1

    # Upload all chunks to GCS and remove them from local memory
    for i in range(chunk_count):
        chunk_filename = f'chunk_{i}_{input_filename}'
        upload_to_gs(BUCKET, chunk_filename, chunk_filename)
        os.remove(chunk_filename)

    for i in range(chunk_count):
        chunk_filename = f'chunk_{i}_{input_filename}'
        
        # Download the chunk from GCS
        download_from_gs(BUCKET, chunk_filename, chunk_filename)
        
        try:
            chunk_actual_duration = get_video_duration(chunk_filename)
            processed_chunk_filename = f'processed_chunk_{i}_{input_filename}'
            await upgrade_audio_video(chunk_filename, processed_chunk_filename, chunk_duration)

            # Upload the processed chunk to GCS
            upload_to_gs(BUCKET, processed_chunk_filename, processed_chunk_filename)

            # Remove the local processed chunk file and original chunk file to free up memory
            os.remove(chunk_filename)
            os.remove(processed_chunk_filename)

        except:
            break

    return chunk_count

async def concatenate_chunks(num_chunks, input_filename, output_filename):
    with open('concat_list.txt', 'w') as f:
        for i in range(num_chunks):
            chunk_filename = f'processed_chunk_{i}_{input_filename}'

            # Download the processed chunk from GCS
            download_from_gs(BUCKET, chunk_filename, chunk_filename)

            f.write(f"file '{chunk_filename}'\n")

    upload_to_gs(BUCKET, 'concat_list.txt', 'concat_list.txt')

    os.system(f'ffmpeg -f concat -i concat_list.txt -c copy {output_filename}')

    upload_to_gs(BUCKET, output_filename, 'concat_chunks_' + output_filename)

    # Clean up the downloaded processed chunks
    for i in range(num_chunks):
        chunk_filename = f'processed_chunk_{i}_{input_filename}'
        os.remove(chunk_filename)

async def process_video(input_filename, output_filename, duration_seconds):
    chunk_duration = 5  # in seconds
    num_chunks = await split_video(input_filename, chunk_duration, output_filename, duration_seconds)
    await concatenate_chunks(num_chunks, input_filename, output_filename)
    codec_output_filename = "codec_" + output_filename
    command = f"ffmpeg -i {output_filename} -vcodec libx264 -acodec aac {codec_output_filename}"
    await run_ffmpeg_command(command)

    upload_to_gs(BUCKET, output_filename, output_filename)
    upload_to_gs(BUCKET, codec_output_filename, codec_output_filename)

    os.remove(codec_output_filename)

    return output_filename, codec_output_filename

async def async_upgrade_audio_video_wrapper(input_filename, output_filename, duration_seconds):

    final_output_filename, codec_final_output_filename = await process_video(input_filename, output_filename, duration_seconds)
    gc.collect()

def sync_upgrade_audio_video_wrapper(input_filename, output_filename):

    with VideoFileClip(input_filename) as video:
        duration_seconds = video.duration

    download_from_gs(BUCKET, input_filename, input_filename)
    asyncio.run(async_upgrade_audio_video_wrapper(input_filename, output_filename, duration_seconds))
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    args = parser.parse_args()

    sync_upgrade_audio_video_wrapper(args.input_filename, args.output_filename)