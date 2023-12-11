### **Group-7_Finals_Face_Recognition**

_**Author/s**: Hannah Jocelle P. Bacong, Jancene Grace C. Generoso, and Renzell M. Mercado_

### Importing Images from Github and Installing Face_Recognition
    !git clone https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies
    !pip install face_recognition
    %cd MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies

### Encoding Profiles Using Known Face Images
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

### Using Facial Recognition on Live-Action Disney Actors and Actresses
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
![Ar with name](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/e66545dc-f757-4ba9-b917-19bf8d3d7151)

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
![Be with name](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/7ae047c6-cb78-40b6-aef0-6afa0d72da2f)

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
![Ci with name](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/e415ae99-13df-4441-87a3-2509142cca66)

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
![Ja with name](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/57c48bd9-cd76-448e-9243-38e1f179da4d)

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
![Mu with name](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/ab5e85e7-6743-47d9-874d-e3730093b343)

### Other Live-Action Disney Actors and Actresses
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
![U1](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/e52fa5fa-be83-416d-8962-43ee54ed8e9d)

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
![U2](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/62f775d7-73a7-47df-ac6f-7ecbc7065d1f)

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
![U3](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/68f6e667-742c-4b03-abc5-210d85155388)

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
![U4](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/56f89d12-8cb2-4d14-92cf-d6fdb2f193e6)

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
![U5](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/d3177af0-7cac-44f6-8409-6250b8e0b821)

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
![U6](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/9179b852-1951-4555-a2c5-18bbae5b2079)

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
![U7](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/9ac026e7-3906-4693-b148-581b3a63da95)

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
![U8](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/92a73f82-8524-4ab9-a4d4-5c523817ba80)

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
![U9](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/846150c0-9812-400b-8357-d50644e98ba1)

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
![U10](https://github.com/renseeel/MX-4101-Group-7_Finals_Face_Recognition_Disney_Movies/assets/143622288/c9150224-e833-4e33-81d8-42564190178d)








