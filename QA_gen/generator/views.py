from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django import forms
from django.views.generic.list import ListView
import pandas as pd

from .models import Document, QAC
from .vae import generator



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
            save = 'media/documents/result.csv'
            results = generator.QA_generation(target, model, save)
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


def qac_del(request, qac_idx):
    qacs = QAC.objects.all()
    dics = []
    local_idx = 0
    for qac in qacs:
        if qac.idx == qac_idx:
            qac.delete()
        else:
            qac.idx = local_idx
            qac.save()
            local_idx += 1
            dic = {}
            # dic['context'] = qac.context
            dic['answer'] = qac.answer
            dic['question'] = qac.question
            dics.append(dic)
    # csvを更新
    save = 'media/documents/result.csv'
    df = pd.DataFrame(dics, columns=['question', 'answer'])
    df.to_csv(save, encoding='cp932')

    return redirect('generator:gen_result')
