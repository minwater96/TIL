### TIL

- django에서 views 함수를 작성할때 중복되는 명령어가 생기게 되는데, 그때 중복되는 명령어 1개를 지우고 나머지를 밖으로 뺴주어 모두 적용될 수 있도록 만들어 줌으로 간략한 함수를 작성해주는 것

 ❗django code 작성의 기본적인 규정으로 잘 신경써서 작성해줘야함

### 예시코드

```python
def create(request):
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        
        if form.is_valid(): # 데이타를 제대로 입력했는지 확인
            form.save() # True값 저장
            return redirect('articles:index')
        # 밑 else문의 내용과 중복되어 제거    
				#else:
				#		context = {
				#				'form'= form,
				#		}
				#		return render(render, 'form.html', context)
    else:
        form = ArticleForm()
        
		# 위 제거된 else문의 수행을 위해 들여쓰기 수행
    context = {
        'form': form,
    }

    return render(request, 'form.html', context)
```

#내맘대로TIL챌린지 #동아일보 #미디어 프론티어 #글로벌소프트웨어캠퍼스 #GSC신촌
글로벌소프트웨어캠퍼스와 동아일보가 함께 진행하는 챌린지입니다. 