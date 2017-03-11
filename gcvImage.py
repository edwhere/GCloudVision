import argparse
import base64

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials


def get_args():
    """
    Collect command-line arguments and return the argument values
    :return: A set of three values: image file name (and path), the operation mode, and the max number of results
    """
    parser = argparse.ArgumentParser(description='Call the Google Vision API to perform image analysis')

    # Add arguments
    parser.add_argument('-i', '--image', type=str, help='image file name', required=True)
    parser.add_argument('-m', '--mode', type=str,
                        help='analysis mode: all, faces, landmark, labels, logos, text, or web', required=True)
    parser.add_argument('-r', '--results', type=int, help='max number of results (default is 5)', default=5)

    # Array for all arguments passed to script
    args = parser.parse_args()

    return args.image, args.mode, args.results


def request_labels(photo_file, max_results=5):
    """
    Request the Google service to analyze an image and return labels (i.e. tags identifying objects in an image)
    :param photo_file: The filename (or path) of the image in a local directory
    :param max_results: The requested maximum number of results
    :return: A list of tuples where each tuple includes a label and a confidence score. The list contains up to
    max_results number of elements
    """
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)

    with open(photo_file, 'rb') as phf:
        image_content = base64.b64encode(phf.read())

        service_request = service.images().annotate(body={
            'requests': [{'image': {'content': image_content.decode('UTF-8')},
                          'features': [{'type': 'LABEL_DETECTION', 'maxResults': max_results}]
                          }]
        })

        response = service_request.execute()

        try:
            label_list = response['responses'][0]['labelAnnotations']
            labels = map(lambda s: (s.get('description', 'no_description'),
                                    s.get('score', 'no_score')), label_list)
            return labels

        except KeyError:
            return []


def request_text(photo_file, max_results=5):
    """
    Request the Google service to find text in an image
    :param photo_file: The filename (or path) of the image in a local directory
    :param max_results: The requested maximum number of results
    :return: A list of text entries found in the image

    Note:  The argument max_results does not modify the number of results for text detection
    """
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)

    with open(photo_file, 'rb') as phf:
        image_content = base64.b64encode(phf.read())

        service_request = service.images().annotate(body={
            'requests': [{'image': {'content': image_content.decode('UTF-8')},
                          'features': [{'type': 'TEXT_DETECTION', 'maxResults': max_results}]
                          }]
        })

        response = service_request.execute()

        text_list = response['responses'][0].get('textAnnotations', None)

        if text_list is None:
            return []
        else:
            text_vec = map(lambda s: s['description'].strip().strip('\n'), text_list)
            return text_vec


def request_faces(photo_file, max_results=5):
    """
    Request the Google service to find faces in an image
    :param photo_file: The filename (or path) of the image in a local directory
    :param max_results: The requested maximum number of results
    :return: A list of JSON objects where each object describes a face. The JSON object includes the following
    elements:
        box: A list of four tuples describing the box coordinates for a face in the picture
        score: A confidence score for face detection
        joy: One of the variables in sentiment analysis of the face
        sorrow: One of the variables in sentiment analysis of the face
        anger: One of the variables in sentiment analysis of the face
        surprise: One of the variables in sentiment analysis of the face
    """

    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)

    with open(photo_file, 'rb') as phf:
        image_content = base64.b64encode(phf.read())

        service_request = service.images().annotate(body={
            'requests': [{'image': {'content': image_content.decode('UTF-8')},
                          'features': [{'type': 'FACE_DETECTION', 'maxResults': max_results}]
                          }]
        })

        response = service_request.execute()

        faces_list = response['responses'][0].get('faceAnnotations', None)

        if faces_list is None:
            return []
        else:
            face_features = []
            for face in faces_list:
                score = face["detectionConfidence"]
                joy = face["joyLikelihood"]
                sorrow = face["sorrowLikelihood"]
                surprise = face["surpriseLikelihood"]
                anger = face["angerLikelihood"]
                vertices = face["boundingPoly"]["vertices"]
                vert_list = map(lambda el: (el['x'], el['y']), vertices)
                face_obj = {'score': score,
                            'joy': joy, 'sorrow': sorrow, 'surprise': surprise,
                            'anger': anger, 'box': vert_list}
                face_features.append(face_obj)

            return face_features


def request_logos(photo_file, max_results=5):
    """
    Request the Google service to detect the presence of logos in an image
    :param photo_file: The filename (or path) of the image in a local directory
    :param max_results: The requested maximum number of results
    :return: A list of tuples where each tuple has text identifying the detected logo and a confidence score
    """
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)

    with open(photo_file, 'rb') as phf:
        image_content = base64.b64encode(phf.read())

        service_request = service.images().annotate(body={
            'requests': [{'image': {'content': image_content.decode('UTF-8')},
                          'features': [{'type': 'LOGO_DETECTION', 'maxResults': max_results}]
                          }]
        })

        response = service_request.execute()

        logo_list = response['responses'][0].get('logoAnnotations', None)

        if logo_list is None:
            return []
        else:
            logo_features = map(lambda s: (s.get("description", "no_description"),
                                           s.get("score", 'no_score')), logo_list)
            return logo_features


def request_landmarks(photo_file, max_results=5):
    """
    Request the Google service to detect the presence of landmarks in an image
    :param photo_file: The filename (or path) of the image in a local directory
    :param max_results: The requested maximum number of results
    :return: A list of tuples where each tuple has text identifying the detected landmark and a confidence score
    """
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)

    with open(photo_file, 'rb') as phf:
        image_content = base64.b64encode(phf.read())

        service_request = service.images().annotate(body={
            'requests': [{'image': {'content': image_content.decode('UTF-8')},
                          'features': [{'type': 'LANDMARK_DETECTION', 'maxResults': max_results}]
                          }]
        })

        response = service_request.execute()

        landmark_list = response['responses'][0].get('landmarkAnnotations', None)

        if landmark_list is None:
            return []
        else:
            landmarks = map(lambda s: (s.get("description", 'no_description'),
                                       s.get("score", 'no_score')), landmark_list)
            return landmarks


def request_web_entities(photo_file, max_results=5):
    """
    Request the Google service to detect related web entities and web pages
    :param photo_file: The filename (or path) of the image in a local directory
    :param max_results: The requested maximum number of results
    :return: This function returns two lists. The first is a list of web entities. The second is a list of related
    web pages. Each list is a collection of tuples. For web entities, each tuple includes the entity description,
    the entity mid, and the entity score. For web pages, each tuple includes the page url and the score.
    """
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)

    with open(photo_file, 'rb') as phf:
        image_content = base64.b64encode(phf.read())

        service_request = service.images().annotate(body={
            'requests': [{'image': {'content': image_content.decode('UTF-8')},
                          'features': [{'type': 'WEB_DETECTION', 'maxResults': max_results}]
                          }]
        })

        response = service_request.execute()

        web_detection = response['responses'][0].get('webDetection', None)

        if web_detection is None:
            return [], []
        else:
            web_entities = web_detection.get('webEntities', None)
            web_pages = web_detection.get('pagesWithMatchingImages', None)

            # Add  s.get('entityId', 'no_entityId') in order to see the mid for these entities
            if web_entities is not None:
                entities = map(lambda s: (s.get('description', 'no_description'),
                                          s.get('score', 'no_score')), web_entities)
            else:
                entities = []

            if web_pages is not None:
                pages = map(lambda s: (s.get('url', 'no_url'), s.get('score', 'no_score')), web_pages)
            else:
                pages = []

            return entities, pages


if __name__ == "__main__":
    # Get image file name and operation mode
    image, mode, mxres = get_args()

    if mode in ['labels', 'all']:
        results = request_labels(image, max_results=mxres)
        print "\n-------- labels -----------------"
        print "Number of labels: ", len(results)
        for i, res in enumerate(results):
            print str(i+1) + ") " + str(res)

    if mode in ['text', 'all']:
        results = request_text(image, max_results=mxres)
        print "\n-------- text -----------------"
        print "Number of text items: ", len(results)
        for i, res in enumerate(results):
            print str(i+1) + ") " + str(res)

    if mode in ['faces', 'all']:
        results = request_faces(image, max_results=mxres)
        print "\n--------- faces ---------------"
        print "Number of faces: ", len(results)
        for i, res in enumerate(results):
            print "\nface " + str(i+1) + ": "
            print res

    if mode in ['logos', 'all']:
        results = request_logos(image, max_results=mxres)
        print "\n--------- logos ---------------"
        print "Number of logos: ", len(results)
        for i, res in enumerate(results):
            print str(i+1) + ") " + str(res)

    if mode in ['landmarks', 'all']:
        results = request_landmarks(image, max_results=mxres)
        print "\n--------- landmarks ---------------"
        print "Number of landmarks: ", len(results)
        for i, res in enumerate(results):
            print str(i+1) + ") " + str(res)

    if mode in ['web', 'all']:
        results = request_web_entities(image, max_results=mxres)
        print "\n--------- web entities and pages ---------------"
        print 'Number of web entities: ', len(results[0])
        for i, res in enumerate(results[0]):
            print str(i+1) + ") " + str(res)

        print "\nNumber of web pages: ", len(results[1])
        for i, res in enumerate(results[1]):
            print str(i+1) + ") " + str(res)
