from django.http import JsonResponse
from .diabetics_symptoms import predict_use_symptoms , cat_col_name
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def symptoms_model(request):
    if request.method == 'POST':
        print("POST request")
        sym_data = [request.POST.get(i) for i in cat_col_name() ]
        pred = predict_use_symptoms(sym_data)
        data = {'message': pred}
        return JsonResponse(data)
    else:
        return JsonResponse({'message': 'This endpoint only accepts POST requests.'})