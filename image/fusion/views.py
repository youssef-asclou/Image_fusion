from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login

from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
#pip install <nom_de_la_bibliotheque> --proxy http://adresse_du_proxy:port
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import math
import torchvision.transforms as transforms
def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']

        # Vérifier si les mots de passe correspondent
        if password != password2:
            messages.error(request, "Les mots de passe ne correspondent pas.")
            return render(request, 'fusion/signup.html')

        # Vérifier si le nom d'utilisateur est déjà pris
        if User.objects.filter(username=username).exists():
            messages.error(request, "Ce nom d'utilisateur est déjà pris.")
            return render(request, 'fusion/signup.html')

        # Vérifier si l'email est déjà pris
        if User.objects.filter(email=email).exists():
            messages.error(request, "Cet email est déjà utilisé.")
            return render(request, 'fusion/signup.html')

        # Créer un nouvel utilisateur
        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()

        # Rediriger vers la page de connexion
        messages.success(request, "Compte créé avec succès ! Vous pouvez maintenant vous connecter.")
        return redirect('login')

    return render(request, 'fusion/signup.html')
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return render(request, 'fusion/fusion.html')  # Redirige l'utilisateur vers la page d'accueil après la connexion
        else:
            # Si les identifiants sont incorrects, affiche un message d'erreur
            return render(request, 'fusion/login.html', {'error': 'Nom d\'utilisateur ou mot de passe incorrect.'})

    return render(request, 'fusion/login.html')



# Modèle amélioré pour la fusion d'images
class ImprovedIFCNN(nn.Module):
    def __init__(self):
        super(ImprovedIFCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fusion_conv = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(16, 3, kernel_size=1, padding=0)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_features(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        return out

    def forward(self, x1, x2):
        out_left = self.forward_features(x1)
        out_right = self.forward_features(x2)
        fusion_in = torch.cat([out_left, out_right], dim=1)
        weights = self.fusion_conv(fusion_in)
        weights = F.softmax(weights, dim=1)
        fused_features = weights[:, 0:1, :, :] * out_left + weights[:, 1:2, :, :] * out_right
        out_final = self.conv_out(fused_features)
        return out_final

# Fonction pour charger le modèle
def load_model(model_path, device):
    model = ImprovedIFCNN()  # Assurez-vous que le modèle est bien défini ici
    # Ajouter map_location pour forcer le chargement sur le CPU si CUDA n'est pas disponible
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()  # Mettre le modèle en mode évaluation
    return model

# Fonction pour traiter et prédire l'image fusionnée
def test_model_on_images(model, left_img, right_img, device):
    # Transformation de l'image (redimensionnement et normalisation)
    transform = transforms.ToTensor()

    left_pil = Image.fromarray(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    right_pil = Image.fromarray(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))

    left_tensor = transform(left_pil).unsqueeze(0).to(device)  # Ajouter la dimension du batch
    right_tensor = transform(right_pil).unsqueeze(0).to(device)

    # Passer les images à travers le modèle
    with torch.no_grad():
        fused_img_tensor = model(left_tensor, right_tensor)
    
    # Convertir la sortie en image
    fused_img = fused_img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H,W,C]
    fused_img = np.clip(fused_img, 0, 1) * 255  # Convertir [0,1] à [0,255]
    fused_img = fused_img.astype(np.uint8)

    return fused_img

# Vérification si CUDA est disponible, sinon utiliser CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('C:/Users/you-asc/IdeaProjects/Image_Fusion/image/models/improved1_ifcnn_model_color (1).pth', device)

@csrf_exempt
def fusion_api(request):
    if request.method == 'POST':
        # Récupère les fichiers envoyés
        file1 = request.FILES.get('image1')
        file2 = request.FILES.get('image2')

        if file1 and file2:
            # Charger les images depuis les fichiers envoyés
            imgA = Image.open(file1).convert('RGB')
            imgB = Image.open(file2).convert('RGB')

            # Convertir les images en tableau numpy
            imgA = np.array(imgA)
            imgB = np.array(imgB)

            # Fusionner les images en utilisant votre modèle
            fused_img = test_model_on_images(model, imgA, imgB, device)

            # Sauvegarder l'image fusionnée dans un buffer mémoire
            _, buffer = cv2.imencode('.png', fused_img)
            fused_base64 = base64.b64encode(buffer).decode('utf-8')

            return JsonResponse({'status': 'ok', 'fused_image': fused_base64})
        else:
            return JsonResponse({'status': 'error', 'message': 'Images manquantes'}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)
