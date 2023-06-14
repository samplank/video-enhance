# Video and Audio Enhancer

This project lets you enhance low-quality videos to make them ready for sharing on social media. It's the open source version of https://www.dubb.media/enhance. 

## How to Run 

1. You will need Python installed on your system. You will also need an account with [Replicate](https://replicate.com/), [DolbyIO](https://dolby.io/products/enhance/), and [Google Cloud](https://cloud.google.com/).

2. Clone this repository:
   ```bash
   git clone https://github.com/samplank/video-enhance.git
   cd video-enhance
Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```
   
Set up the environment variables:
You will need to set up several environment variables to run this script. You can do this by creating a .env file in the root of the project or setting them from the command line:

```bash
DOLBY_APP_KEY=YOUR_DOLBY_APP_KEY
DOLBY_APP_SECRET=YOUR_DOLBY_APP_SECRET
REPLICATE_API_TOKEN=YOUR_REPLICATE_API_TOKEN
REPLICATE_ENHANCE_MODEL=YOUR_REPLICATE_ENHANCE_MODEL
BUCKET=YOUR_GOOGLE_CLOUD_BUCKET_NAME
GOOGLE_APPLICATION_CREDENTIALS=PATH_TO_YOUR_SERVICE_ACCOUNT_JSON
```
Run the script like this:

```
python enhance_video_audio.py input.mp4 output.mp4
```
