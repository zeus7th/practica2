# Importar módulos necesarios
import streamlit as st  # Biblioteca para crear aplicaciones web
from PIL import Image
import json
import requests
import os
import io

st.title('Streamlit App')

col1, col2 = st.columns(2)
with col1:   
    
    col1.markdown('### Generación de Imagenes')
    input_text = st.text_input('Ingrese algún Texto')
    if st.button('Generar Imagen'):
        url =  "https://stablediffusionapi.com/api/v4/dreambooth"     
        payload = json.dumps({  
        "key":  "DcFrfw2jKozWaSumHty1f9R6EMtExTKpX7XUbiw2ZARGK3crPr2cqTBFYu0L",  
        "model_id":  "juggernaut-xl-v5",         
        "prompt": input_text,  
        "negative_prompt":  "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",  
        "height":  "512",  
        "samples":  "1",  
        "num_inference_steps":  "30",  
        "safety_checker":  "no",  
        "enhance_prompt":  "yes",  
        "seed":  None,  
        "guidance_scale":  7.5,  
        "multi_lingual":  "no",  
        "panorama":  "no",  
        "self_attention":  "no",  
        "upscale":  "no",  
        "embeddings":  "embeddings_model_id",  
        "lora":  "lora_model_id",  
        "webhook":  None,  
        "track_id":  None  
        })  
    
        headers =  {  
        'Content-Type':  'application/json'  
        }  
    
        response = requests.request("POST", url, headers=headers, data=payload)  
    
        #print(response.text)
        data=response.json()

        if data.get('status') == 'error':
            # Obtener y mostrar el mensaje de error si está presente
            error_message = data.get('message', 'No se encontró ningún mensaje de error en la respuesta JSON.')
            st.write(error_message)
        else:
            urlimagen=response.json()['future_links'][0]
            response.json()['future_links'][0]
            st.image(urlimagen, caption='Imagen Mostrada')


with col2:
    col2.markdown(' ### Clasificación de Imagenes')
    uploaded_image = st.file_uploader("Selecciona una imagen",type=["jpg","jpeg","png"])
    if uploaded_image is not None:
        #image = Image.open(uploaded_image)
        
        # Convertir la imagen a un objeto de imagen de Pillow
        try:
            image_bytes = uploaded_image.read()
            image = Image.open(io.BytesIO(image_bytes))
        except (IOError, OSError) as e:
            st.write(f"Error al abrir la imagen: {e}")
            st.stop()

        st.image(image,caption="Imagen cargada",use_column_width=True)

    if st.button('Clasificar Imagen'):
        url = 'http://127.0.0.1:8000/computer-vision'
        try:        
            #image_bytes = uploaded_image.read()
            #image = Image.open(io.BytesIO(image_bytes.read()))
            files = {'file': ('image.jpg',image_bytes)}
            #headers = {'Content-Type': 'multipart/form-data'}
            response = requests.post(url=url,files=files)
            if response.status_code==200:
                predicted_class = response.json().get('predicted_class', '')
                st.write(f"La imagen corresponde a: {predicted_class}")
                print(response.text)
                
            else:
                st.write(f"Error al clasificar la imagen. Códido de estado: {response.status_code}")
        except (IOError,OSError) as e:
            st.write(f"Error al abrir la imagen: {e}")


        
