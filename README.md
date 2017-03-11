# Image Analysis Using the Google Cloud Vision API

## Introduction

The Cloud Vision API is an image analysis service that is part of Google's Machine Learning cloud infrastructure. The engine behind the Cloud Vision API performs image analysis in the following categories: Label detection, Face detection, Text detection, Logo detection, and Landmark detection. For more information access Google's description page:

                         https://cloud.google.com/vision/

The Python software described here uses the Cloud Vision API to perform image analysis in any of the categories. 

### Prerequisites

1. You need to have a Google Cloud Platform (GCP) account. For more information go to: 

        https://cloud.google.com/ for more information.

2. Set up a GCP project and enable the use of the Cloud Vision API for your project

3. Create service credentials for the project as described in:

        https://cloud.google.com/vision/docs/common/auth

    Note: Google provides two methods for authentication: API keys or using service credentials. The code in this page uses the latter.

4. Install the Google API Client Libraries (Python). In my case I created a conda environment and then used pip install. This document has installation instructions:

        https://developers.google.com/api-client-library/python/start/installation

5. The software runs using Python 2.7

### Installing

1. Download the python file
2. If you have set up an environment to run Google API python libraries then activate the environment 

## Running the program

The program runs from command line. Assuming you have an image called myImage.jpg in the same directory as the Python file then the following commands are available: 

python gcvImage.py -h                                       <-  Displays help information

python gcvImage.py -i myImage.jpg -m labels -r 6            <-  Request to detect objects with a maximum of 6 results

python gcvImage.py -i myImage.jpg -m text                   <-  Request to detect text in an image 

python gcvImage.py -i myImage.jpg -m faces -r 6             <-  Request to detect faces with a maximum of 6 results

python gcvImage.py -i myImage.jpg -m logos -r 6             <-  Request to detect logos with a maximum of 6 results

python gcvImage.py -i myImage.jpg -m landmarks -r 6         <-  Request to detect landmarks with a maximum of 6 results

python gcvImage.py -i myImage.jpg -m web -r 6               <-  Request to detect related web entities and web pages with a maximum of 6 results

python gcvImage.py -i myImage.jpg -m all -r 6               <-  Request to run all modes (objects, text, faces, logos, landmarks, web entities)

## Authors

Edwin Heredia 

https://github.com/edwhere

http://www.cometglow.com


## License

This project is licensed under the MIT License (Expat version)

Copyright (c) 2017 - 2020 Edwin Heredia
 
 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





