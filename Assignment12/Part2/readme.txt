"dog01.jpg3450": {						— Each image is a son object with the name being concatenation of uploaded image name and the size in bytes
        "filename": "dog01.jpg",			— File name of the uploaded image
        "size": 3450,						— Size of the image in bytes
        "regions": [						— Annotated regions represented as a son array since there can be multiple annotations in the same image
            {
                "shape_attributes": {			— Describe the properties of the shape used for annotating
                    "name": "rect",				— In this case rectangular shape was chosen
                    "x": 45,						— x coord of the starting pixel of annotation
                    "y": 31,						— x coord of the starting pixel of annotation
                    "width": 39,				— width of the annotated region in pixels
                    "height": 70				— height of the annotated region in pixels
                },								— together x, y, width, height can be used to find where to draw the bounding box and with what dimension
                "region_attributes": {			— Annotation properties along with the values provided during annotation
                    "name": "dog1",
                    "class": "dog"
                }
            }        
	],
        "file_attributes": {					— Generic attributes
            "caption": "",					— Caption, if any, of the chosen image
            "public_domain": "no",			— 
            "image_url": ""
        }
