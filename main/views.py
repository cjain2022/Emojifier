from django.shortcuts import render
from emojifier_DeepLearningCode import predict
# Create your views here.
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