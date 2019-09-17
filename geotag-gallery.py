import argparse
import glob
import os
import time

import numpy as np
import PIL.ExifTags
import PIL.Image
import requests
from dateutil.parser import parse
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array
from lxml import etree
from pykml.factory import KML_ElementMaker as KML
from textblob import TextBlob

KML_POPUP_WIDTH = 400
REVERSE_GEOCODE_URL = 'https://nominatim.openstreetmap.org/reverse'
SLEEP_BEFORE_REQUEST = 2
MODEL = ResNet50(weights="imagenet")
OUTPUT_FILE = 'photos.kml'

parser = argparse.ArgumentParser(description='Create placemarks from photos and export to KML for Google Earth.')
parser.add_argument('--folder', action='store', dest='folder', help='Pick a full path to geotagged image folder.')
parser.add_argument('--language', action='store', dest='language', default='en', help='Pick a language code.')


locations = []


def prepare_image(image, target):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image


def predict(image):
    image = prepare_image(image, target=(224, 224))
    predictions = MODEL.predict(image)
    results = imagenet_utils.decode_predictions(predictions)
    names = [result[1] for result in results[0][:2]]
    return u'/'.join(names)


def retry(url, times=5, sleep=5, **kwargs):
    response = None
    while times:
        response = requests.get(url, **kwargs)
        if response.status_code != 200:
            response = None
            times -= 1
            time.sleep(sleep)
        else:
            break
    return response


def get_local_location(latitude, longitude):
    for location in locations:
        if location['latitude'] == latitude and location['longitude'] == longitude:
            return location


def reverse_geocode(latitude, longitude):
    """Return display name for coordinates by contacting the openstreetmap API.
    Because the API requires that there's a cache running and that requests are at least a second
    apart, it has been implemented to retry.
    """
    payload = {
        'lat': latitude,
        'lon': longitude,
        'format': 'json',
        'zoom': 18,
        'addresdetails': 1
    }
    location = get_local_location(latitude, longitude)
    if location:
        return location['display_name']
    time.sleep(SLEEP_BEFORE_REQUEST)
    try:
        response = retry(REVERSE_GEOCODE_URL, params=payload)
        display_name = response.json()['display_name']
    except (requests.exceptions.ConnectionError, AttributeError):
        display_name = u'Unknown'
    locations.append({'latitude': latitude, 'longitude': longitude, 'display_name': display_name})
    return display_name


def convert_to_decimal(degrees, minutes, seconds, direction):
    coordinate = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction == 'W' or direction == 'S':
        coordinate *= -1
    return coordinate


def build_kml(folder, kml_folder):
    when = u'Photographed at'
    where = u'near'
    if args.language != 'en':
        when = TextBlob(when).translate(to=args.language)
        where = TextBlob(where).translate(to=args.language)
    for path in glob.glob(os.path.join(folder, '*.jp*')):
        image = PIL.Image.open(path)
        if args.language != 'en':
            name = TextBlob(predict(image)).translate(to=args.language) or os.path.split(path)[-1]
        else:
            name = predict(image) or os.path.split(path)[-1]
        exif = {PIL.ExifTags.TAGS[key]: value for key, value in image._getexif().items() if key in PIL.ExifTags.TAGS}
        try:
            degrees = float(exif['GPSInfo'][2][0][0]) / float(exif['GPSInfo'][2][0][1])
            minutes = float(exif['GPSInfo'][2][1][0]) / float(exif['GPSInfo'][2][1][1])
            seconds = float(exif['GPSInfo'][2][2][0]) / float(exif['GPSInfo'][2][2][1])
            direction = exif['GPSInfo'][1]
            latitude = convert_to_decimal(degrees, minutes, seconds, direction)
            degrees = float(exif['GPSInfo'][4][0][0]) / float(exif['GPSInfo'][4][0][1])
            minutes = float(exif['GPSInfo'][4][1][0]) / float(exif['GPSInfo'][4][1][1])
            seconds = float(exif['GPSInfo'][4][2][0]) / float(exif['GPSInfo'][4][2][1])
            direction = exif['GPSInfo'][3]
            longitude = convert_to_decimal(degrees, minutes, seconds, direction)
            relative_path = path.replace(args.folder, '')
            pm = KML.Placemark(
                KML.name(name),
                KML.description(
                    u'<h1>{name}</h1><p>{desc}</p><p><img src="{path}" width="{width}"></p>'.format(
                            name=name,
                            desc=u'{} {} {} {}'.format(when, exif['DateTimeDigitized'], where, reverse_geocode(latitude, longitude)),
                            path=relative_path,
                            width=KML_POPUP_WIDTH
                    )
                ),
                KML.Point(
                    KML.coordinates('{},{}'.format(longitude, latitude))
                ),
                KML.TimeStamp(
                    KML.when(parse(exif['DateTimeDigitized']).isoformat())
                )
            )
            kml_folder.append(pm)
        except KeyError:
            continue


def main():
    kml_folder = KML.Folder()
    for dir_name, sub_dirs, files in os.walk(args.folder):
        build_kml(dir_name, kml_folder)
    with open(os.path.join(args.folder, OUTPUT_FILE), 'w') as kml:
        kml.write(etree.tostring(kml_folder, pretty_print=True))


if __name__ == '__main__':
    args = parser.parse_args()
    main()
