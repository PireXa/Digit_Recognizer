from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from PIL import Image
import io
import base64
import json
import os
from django.conf import settings
from .AI_model.create_train import create_train_model, predict
import numpy as np

# Create your views here.
def canvas(request):
    return render(request, 'draw.html')

def submit_drawing(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            width, height = 400, 400  # Assuming these are your canvas dimensions
            image = Image.new('RGBA', (width, height))

            pixels = image.load()
            for i in range(height):
                for j in range(width):
                    index = (i * width + j) * 4
                    r, g, b, a = data[index:index + 4]
                    if a == 0:
                        r, g, b, a = 255, 255, 255, 255
                    pixels[j, i] = (r, g, b, a)

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            # Resize the image to the expected input size (e.g., 28x28)
            eval_image = image.resize((28, 28))
            # eval_image.show()
            # image.show()
            
            # # Convert to grayscale if needed
            eval_image = eval_image.convert('L')  # Convert to grayscale
            # eval_image.show()
            
            # # Convert the image to a NumPy array
            eval_image_array = np.array(eval_image).astype('float32') / 255  # Normalize
            eval_image_array = 1 - eval_image_array  # Invert the image
            print(eval_image_array)
            
            # # Reshape the image for model input
            eval_image_array = eval_image_array.reshape((1, 28, 28, 1))  # Add batch dimension

            
            # # Make predictions
            predicted_class = predict(eval_image_array)
            print('Predicted class:', predicted_class)

            return JsonResponse({'status': 'success', 'image': img_base64, 'predicted_class': int(predicted_class)})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

def train_model(request):
    # if not os.path.exists(os.path.join(settings.BASE_DIR, 'recognition_app', 'AI_model', 'trained_models', 'mnist_model.h5')):
    create_train_model()
    return JsonResponse({'status': 'success', 'message': 'Model trained successfully'})
