from django.shortcuts import render, redirect
from django.http import HttpResponse
from django import forms
from django.views.generic.list import ListView
from .models import Document, QAC
from .vae import generator
import glob


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('document', )


# Create your views here.
def doc_register(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            docs = Document.objects.all()
            for doc in docs:
                doc.delete()
            form.save()

            target = 'media/' + Document.objects.all()[0].document.name
            model = 'media/model/best_f1_model.pt'
            results = generator.QA_generation(target, model)
            qacs = QAC.objects.all()
            for qac in qacs:
                qac.delete()
            for idx, res in enumerate(results):
                qac = QAC()
                qac.idx = idx
                qac.question = res['question']
                qac.answer = res['answer']
                qac.context = res['context']
                qac.save()

            return redirect('generator:gen_result')
    else:
        form = DocumentForm()
    return render(request, 'generator/doc_register.html', {
        'form': form
    })


class ResultList(ListView):
    context_object_name = 'qacs'
    template_name = 'generator/result.html'
    paginate_by = 10

    def get(self, request, *args, **kwargs):
        self.object_list = QAC.objects.all()
        context = self.get_context_data(object_list=self.object_list)
        return self.render_to_response(context)
