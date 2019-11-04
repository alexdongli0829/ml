from collections import deque

q = deque([1,2,3])
for i in range (10,20):
	print(i)
	q.append(i)
	improv = 0
	if len(q) > 5:
		q.popleft()
		for j in q:
			improv =improv + j
		print(q)
		if improv > 20:
			print(q)
			break
