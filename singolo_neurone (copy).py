import numpy as np
import matplotlib.pyplot as plt
#dati
#lato_a = np.array([3,7,10,1])
#lato_b = np.array([5,4,20,1])
#perimetro_vero = np.array([16,22,60,4])

lato_a = np.array([3,1,4])
lato_b = np.array([5,1,2])
perimetro_vero = np.array([16,4,12])

epoche = 3000
tasso_apprendimento = 0.006

#addestramento
#inizializzazione pesi e bias
p_a = 1.
p_b = 1.
off = 1.

# Lists to store values for plotting
p_a_values = []
p_b_values = []
off_values = []

for epoch in range(epoche):

    #forward propagation
    y = lato_a*p_a +lato_b*p_b + off 

    #calcolo errore
    loss = np.mean((y-perimetro_vero)**2)

    #backward propagation
    grad_y = 2*(y-perimetro_vero)   #d loss / d y

    grad_p_a = np.mean(grad_y*lato_a)
    grad_p_b = np.mean(grad_y*lato_b)

    grad_off = np.mean(grad_y)

    #aggiornamento pesi
    p_a = p_a - tasso_apprendimento*grad_p_a
    p_b = p_b - tasso_apprendimento*grad_p_b
    off = off - tasso_apprendimento*grad_off

    # Append values for plotting
    p_a_values.append(p_a)
    p_b_values.append(p_b)
    off_values.append(off)

    print(f'Epoch {epoch}, Loss: {loss}')
    print(p_a)
    print(p_b)
    print(off)
    #if epoch % 10 == 0:
    #    print(f'Epoch {epoch}, Loss: {loss}')


# # Effettua una previsione con nuovi dati
# nuovo_lato_a = 5.0
# nuovo_lato_b = 8.0
# previsione = nuovo_lato_a*p_a + nuovo_lato_b*p_b + off
# print(f'Perimetro previsto per il rettangolo con lati {nuovo_lato_a} e {nuovo_lato_b}: {previsione}')
# print(p_a)
# print(p_b)
# print(off)

# Plotting
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(epoche), p_a_values, marker='o')
plt.title('Convergence of p_a')
plt.xlabel('Epochs')
plt.ylabel('p_a')

plt.subplot(1, 3, 2)
plt.plot(range(epoche), p_b_values, marker='o')
plt.title('Convergence of p_b')
plt.xlabel('Epochs')
plt.ylabel('p_b')

plt.subplot(1, 3, 3)
plt.plot(range(epoche), off_values, marker='o')
plt.title('Convergence of off')
plt.xlabel('Epochs')
plt.ylabel('off')

plt.tight_layout()
plt.show()