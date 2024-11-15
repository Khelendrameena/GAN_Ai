<h1>Video Generation</h1>
<h3>Overview</h3>
<p>This project demonstrates a Generative Adversarial Network (GAN)-based model that generates frames sequentially to create videos. The model takes an input image, generates an output frame, and then reuses the output frame as input for the next iteration to create a sequence of frames. The generated frames are then combined into a video.</p>
<h3>
  Key Features
</h3>
<p>
    &#9679;  Frame-by-Frame Generation: The GAN generates one frame at a time, reusing the previous output as input for the next frame.<br>
  
   &#9679;  Video Generation: Combines generated frames into a complete video.
Custom Dataset Creation: Extracts video frames and processes them for training and testing.<br>
   &#9679;  Sentence-to-Image Conversion: Converts a sentence into an image representation using Word2Vec embeddings.
</p>
