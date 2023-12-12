# <h1 align="center">𝕲𝖗𝖔𝖚𝖕-7_𝕱𝖎𝖓𝖆𝖑𝖘_𝕱𝖆𝖈𝖊_𝕽𝖊𝖈𝖔𝖌𝖓𝖎𝖙𝖎𝖔𝖓</h1>

_**𝓐𝓾𝓽𝓱𝓸𝓻/𝓼**: Hannah Jocelle P. Bacong, Jancene Grace C. Generoso, and Renzell M. Mercado_

### ℑ𝔪𝔭𝔬𝔯𝔱𝔦𝔫𝔤 ℑ𝔪𝔞𝔤𝔢𝔰 𝔣𝔯𝔬𝔪 𝔊𝔦𝔱𝔥𝔲𝔟 𝔞𝔫𝔡 ℑ𝔫𝔰𝔱𝔞𝔩𝔩𝔦𝔫𝔤 𝔉𝔞𝔠𝔢_ℜ𝔢𝔠𝔬𝔤𝔫𝔦𝔱𝔦𝔬𝔫
    !git clone https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies
    !pip install face_recognition
    %cd MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies

### 𝔈𝔫𝔠𝔬𝔡𝔦𝔫𝔤 𝔓𝔯𝔬𝔣𝔦𝔩𝔢𝔰 𝔘𝔰𝔦𝔫𝔤 𝔎𝔫𝔬𝔴𝔫 𝔉𝔞𝔠𝔢 ℑ𝔪𝔞𝔤𝔢𝔰
    import face_recognition
    import numpy as np
    from google.colab.patches import cv2_imshow
    import cv2

    # Creating the encoding profiles
    face_1 = face_recognition.load_image_file("Ariel.jpg")
    face_1_encoding = face_recognition.face_encodings(face_1)[0]

    face_2 = face_recognition.load_image_file("Belle.jpg")
    face_2_encoding = face_recognition.face_encodings(face_2)[0]

    face_3 = face_recognition.load_image_file("Cinderella.jpg")
    face_3_encoding = face_recognition.face_encodings(face_3)[0]

    face_4 = face_recognition.load_image_file("Jasmine.jpg")
    face_4_encoding = face_recognition.face_encodings(face_4)[0]

    face_5 = face_recognition.load_image_file("Mulan.jpg")
    face_5_encoding = face_recognition.face_encodings(face_5)[0]

    known_face_encodings = [
                        face_1_encoding,
                        face_2_encoding,
                        face_3_encoding,
                        face_4_encoding,
                        face_5_encoding
    ]

    known_face_names = [
                    "Ariel",
                    "Belle",
                    "Cinderella",
                    "Jasmine",
                    "Mulan",
    ]

### 𝔘𝔰𝔦𝔫𝔤 𝔉𝔞𝔠𝔦𝔞𝔩 ℜ𝔢𝔠𝔬𝔤𝔫𝔦𝔱𝔦𝔬𝔫 𝔬𝔫 𝔏𝔦𝔳𝔢-𝔄𝔠𝔱𝔦𝔬𝔫 𝔇𝔦𝔰𝔫𝔢𝔶 𝔄𝔠𝔱𝔬𝔯𝔰 𝔞𝔫𝔡 𝔄𝔠𝔱𝔯𝔢𝔰𝔰𝔢𝔰
    file_name = "Ar.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/e66545dc-f757-4ba9-b917-19bf8d3d7151" alt="Ar with name">
</p>

    file_name = "Be.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)
<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/7ae047c6-cb78-40b6-aef0-6afa0d72da2f" alt="Be with name">
</p>

    file_name = "Ci.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)
    
<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/e415ae99-13df-4441-87a3-2509142cca66" alt="Ci with name">
</p>

    file_name = "Ja.jpeg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)
    
<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/57c48bd9-cd76-448e-9243-38e1f179da4d" alt="Ja with name">
</p>

    file_name = "Mu.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)
    
<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/ab5e85e7-6743-47d9-874d-e3730093b343" alt="Mu with name">
</p>


### 𝔒𝔱𝔥𝔢𝔯 𝔏𝔦𝔳𝔢-𝔄𝔠𝔱𝔦𝔬𝔫 𝔇𝔦𝔰𝔫𝔢𝔶 𝔄𝔠𝔱𝔬𝔯𝔰 𝔞𝔫𝔡 𝔄𝔠𝔱𝔯𝔢𝔰𝔰𝔢𝔰
    file_name = "An.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)
    
<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/e52fa5fa-be83-416d-8962-43ee54ed8e9d" alt="U1">
</p>

    file_name = "Me.jpeg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/62f775d7-73a7-47df-ac6f-7ecbc7065d1f" alt="U2">
</p>

    file_name = "Po.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/68f6e667-742c-4b03-abc5-210d85155388" alt="U3">
</p>

    file_name = "Mo.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/56f89d12-8cb2-4d14-92cf-d6fdb2f193e6" alt="U4">
</p>

    file_name = "Ki.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/d3177af0-7cac-44f6-8409-6250b8e0b821" alt="U5">
</p>

    file_name = "Sw.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/9179b852-1951-4555-a2c5-18bbae5b2079" alt="U6">
</p>

    file_name = "Mi.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/9ac026e7-3906-4693-b148-581b3a63da95" alt="U7">
</p>

    file_name = "Al.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/92a73f82-8524-4ab9-a4d4-5c523817ba80" alt="U8">
</p>

    file_name = "Mk.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/846150c0-9812-400b-8357-d50644e98ba1" alt="U9">
</p>

    file_name = "Ri.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
        cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)

<p align="center">
  <img src="https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/c9150224-e833-4e33-81d8-42564190178d" alt="U10">
</p>

### 🅢🅞🅤🅡🅒🅔
https://disneyprincess.fandom.com/wiki/Live-Action_Disney_Princesses






