from django.shortcuts import render
from emojifier_DeepLearningCode import predict
# Create your views here.
from decouple import config
DOWNLOAD = config('DOWNLOAD',cast=bool, default=False)

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
def execute_download():
    # emojifier_DeepLearningCode\embedding\glove.6B.50d.txt
    download_file_from_google_drive('1MTlxFUh2PnT68HKvnUdTa4jpycRFb13O','emojifier_DeepLearningCode/embedding/glove.6B.50d.txt')

from django.http import HttpResponse
def temp_view(request):
    if DOWNLOAD:
        print("ONE TIME DOWNLOAD OF EMBEDDINGS FILE ")
        execute_download()
        return HttpResponse("Download success")
    return HttpResponse("Download was set to false")

def predictEmoji(request):
    if request.method=="GET":
        return render(request,'main/index.html')
    if request.method=="POST":
        text=request.POST['text']
        print(text)
        emoji=predict.predict(text)
        context={
            'emoji':emoji
        }
    response=render(request,'main/index.html',context)
    return response

# from services.Emojifier import predictor
# # Create your views here.
# class Index(TemplateView):
#     template_name='main/index.html'

#     def post(self,request):
#         content=request.POST['content']
#         # print(content)
#         emoji=predictor.predict(content)
#         context={
#             'input_data':content,
#             'emoji':emoji
#         }
#         return render(request,self.template_name,context=context)