from flask import Blueprint, render_template, redirect, url_for, request, flash, session, current_app, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from . import Fuser_collection, Buser_collection, mail, password_reset_token, messages, socketio, posts, app, AI_collection
from flask_mail import Message
import secrets
from datetime import datetime, timedelta
from flask_socketio import join_room, leave_room, send
from bson import json_util
import os
from bson.objectid import ObjectId
from functools import wraps
import uuid
from werkzeug.utils import secure_filename 
import torch
from torchvision import transforms
from PIL import Image
from .model import CropDiseaseModel


model_path = os.path.join(os.path.dirname(__file__), "models", "crop_disease_model.pth")
NUM_CLASSES = 22 
model = CropDiseaseModel(NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_LABELS = ['Cashew anthracnose', 'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust', 'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic', 'Maize fall armyworm', 'Maize grasshoper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight', 'Maize leaf spot', 'Maize streak virus', 'Tomato healthy', 'Tomato leaf blight', 'Tomato leaf curl', 'Tomato septoria leaf spot', 'Tomato verticulium wilt']  # Update with your classes


auth = Blueprint('auth', __name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def send_reset_email(user_email, reset_link):
    msg = Message('Password Reset Request',
                  recipients=[user_email])
    msg.body = f'''To reset your password, visit the following link:{reset_link}. If you did not make this request, simply ignore this email.'''
    mail.send(msg)

@auth.route('/Freset_password', methods=['GET', 'POST'])
def Freset_request():
    if request.method == 'POST':
        email = request.form.get('email')
        user = Fuser_collection.find_one({'email': email})
        if user:

            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(hours=1)
            

            password_reset_token.insert_one({
                'email': email,
                'token': token,
                'expires_at': expires_at
            })
            
            # Send reset email
            reset_link = url_for('auth.Freset_token', token=token, _external=True)
            send_reset_email(email, reset_link)
            flash('Reset link has been sent to your email', 'info')
            return redirect(url_for('auth.Flogin'))
        else:
            flash('Email not found', 'danger')
    return render_template('reset_request.html')

@auth.route('/Breset_password', methods=['GET', 'POST'])
def Breset_request():
    if request.method == 'POST':
        email = request.form.get('email')
        user = Buser_collection.find_one({'email': email})
        if user:

            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(hours=1)
            

            password_reset_token.insert_one({
                'email': email,
                'token': token,
                'expires_at': expires_at
            })
            
            # Send reset email
            reset_link = url_for('auth.Breset_token', token=token, _external=True)
            send_reset_email(email, reset_link)
            flash('Reset link has been sent to your email', 'info')
            return redirect(url_for('auth.Blogin'))
        else:
            flash('Email not found', 'danger')
    return render_template('reset_request.html')

@auth.route('/Freset_password/<token>', methods=['GET', 'POST'])
def Freset_token(token):
    token_data = password_reset_token.find_one({'token': token})
    
    if not token_data or token_data['expires_at'] < datetime.utcnow():
        flash('Invalid or expired token', 'danger')
        return redirect(url_for('auth.reset_request'))
    
    if request.method == 'POST':
        new_password = request.form.get('password')
        hashed_password = generate_password_hash(new_password)
        
        # Update password
        Fuser_collection.update_one(
            {'email': token_data['email']},
            {'$set': {'password': hashed_password}}
        )
        
        # Delete used token
        password_reset_token.delete_one({'token': token})
        
        flash('Your password has been updated!', 'success')
        return redirect(url_for('auth.Flogin'))
    
    return render_template('reset_token.html')

@auth.route('/Breset_password/<token>', methods=['GET', 'POST'])
def Breset_token(token):
    token_data = password_reset_token.find_one({'token': token})
    
    if not token_data or token_data['expires_at'] < datetime.utcnow():
        flash('Invalid or expired token', 'danger')
        return redirect(url_for('auth.reset_request'))
    
    if request.method == 'POST':
        new_password = request.form.get('password')
        hashed_password = generate_password_hash(new_password)
        
        # Update password
        Buser_collection.update_one(
            {'email': token_data['email']},
            {'$set': {'password': hashed_password}}
        )
        
        # Delete used token
        password_reset_token.delete_one({'token': token})
        
        flash('Your password has been updated!', 'success')
        return redirect(url_for('auth.Blogin'))
    
    return render_template('reset_token.html')



@auth.route('/Fsignup', methods=['POST', 'GET'])
def Fsignup():
    if request.method == 'POST':
        firstname=request.form.get('FirstName')
        secondname=request.form.get('SecondName')
        phonenumber=request.form.get('PhoneNumber')
        email=request.form.get('email')
        password=request.form.get('password')
        confirm_password=request.form.get('ConfirmPassword')

        if password != confirm_password:
            flash('password does not match', 'danger')
            return redirect(url_for('auth.Fsignup'))

        if Fuser_collection.find_one({'email': email}):
            flash('Email already exists', 'danger')
            return redirect(url_for('auth.Fsignup'))
        if Fuser_collection.find_one({'phonenumber': phonenumber}):
            flash('PhoneNumber already exists', 'danger')
            return redirect(url_for('auth.Fsignup'))
        hashed_psw=generate_password_hash(password)
        user_data={
            'firstname':firstname,
            'secondname':secondname,
            'phonenumber':phonenumber,
            'email': email,
            'password': hashed_psw,
            'registered_on':datetime.utcnow(),
            'username': firstname + secondname
        }

        Fuser_collection.insert_one(user_data)
        return redirect(url_for('auth.Flogin'))
    return render_template('Fsignup.html')

@auth.route('/Bsignup', methods=['POST', 'GET'])
def Bsignup():
    if request.method == 'POST':
        firstname=request.form.get('FirstName')
        secondname=request.form.get('SecondName')
        phonenumber=request.form.get('PhoneNumber')
        email=request.form.get('email')
        password=request.form.get('password')
        confirm_password=request.form.get('ConfirmPassword')

        if password != confirm_password:
            flash('password does not match', 'danger')
            return redirect(url_for('auth.Fsignup'))

        if Buser_collection.find_one({'email': email}):
            flash('Email already exists', 'danger')
            return redirect(url_for('auth.Fsignup'))
        if Buser_collection.find_one({'phonenumber': phonenumber}):
            flash('PhoneNumber already exists', 'danger')
            return redirect(url_for('auth.Fsignup'))
        hashed_psw=generate_password_hash(password)
        user_data={
            'firstname':firstname,
            'secondname':secondname,
            'phonenumber':phonenumber,
            'email': email,
            'password': hashed_psw,
            'registered_on':datetime.utcnow(),
            'username': firstname + secondname
        }

        Buser_collection.insert_one(user_data)
        return redirect(url_for('auth.Blogin'))
    return render_template('Bsignup.html')

@auth.route('/Flogin', methods=['POST', 'GET'])
def Flogin():
    if request.method == 'POST':
        phonenumber=request.form.get('PhoneNumber')
        password=request.form.get('password')

        user=Fuser_collection.find_one({'phonenumber':phonenumber})
        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            session['email'] = user['email']
            session['phonenumber'] = user['phonenumber']
            session['user_id'] = str(user['_id'])
            return redirect(url_for('auth.Fhome'))
        else:
            flash('Invalid phonenumber or password', 'danger')
            return redirect(url_for('auth.Flogin'))
    return render_template('Flogin.html')

@auth.route('/Blogin', methods=['POST', 'GET'])
def Blogin():
    if request.method == 'POST':
        phonenumber=request.form.get('PhoneNumber')
        password=request.form.get('password')

        user=Buser_collection.find_one({'phonenumber':phonenumber})
        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            session['email'] = user['email']
            session['phonenumber'] = user['phonenumber']
            session['user_id'] = str(user['_id'])
            return redirect(url_for('auth.Bhome'))
        else:
            flash('Invalid phonenumber or password', 'danger')
            return redirect(url_for('auth.Blogin'))
    return render_template('Blogin.html')

@auth.route('/Fhome', methods=['GET'])
def Fhome():
    required_keys = ['username', 'email', 'phonenumber']
    if not all(key in session for key in required_keys):
        return redirect(url_for('auth.Flogin'))
    user_data = {
        'username': session['username'],
        'email': session['email'],
        'phonenumber': session['phonenumber'],
        'user_id':session['user_id']
    }
    all_posts = posts.find().sort('timestamp', -1)
    return render_template('Fhome.html', user=user_data, posts=all_posts)

@auth.route('/Bhome', methods=['POST', 'GET'])
def Bhome():
    required_keys = ['username', 'email', 'phonenumber']
    if not all(key in session for key in required_keys):
        return redirect(url_for('auth.Blogin'))
    user_data = {
        'username': session['username'],
        'email': session['email'],
        'phonenumber': session['phonenumber'],
        'user_id':session['user_id']
    }
    all_posts = posts.find().sort('timestamp', -1)
    return render_template('Bhome.html', user=user_data, posts=all_posts)

@auth.route('/Flogout')
def Flogout():
    session.pop('username', None)
    return redirect(url_for('auth.Flogin'))

@auth.route('/Blogout')
def Blogout():
    session.pop('username', None)
    return redirect(url_for('auth.Blogin'))


@auth.route('/Fusers')
def Finbox():
    if 'username' not in session:
        return redirect(url_for('auth.Flogin'))
    users_list = Buser_collection.find({}, {"username": 1})
    unread_counts = messages.aggregate([
        {"$match": {"receiver": session['username'], "read": False}},
        {"$group": {"_id": "$sender", "count": {"$sum": 1}}}
    ])
    unread_dict = {item['_id']: item['count'] for item in unread_counts}
    
    return render_template('Finbox.html', 
                           users=users_list, 
                           unread_dict=unread_dict)

@auth.route('/Busers')
def Binbox():
    if 'username' not in session:
        return redirect(url_for('auth.Blogin'))
    users_list = Fuser_collection.find({}, {"username": 1})
    unread_counts = messages.aggregate([
        {"$match": {"receiver": session['username'], "read": False}},
        {"$group": {"_id": "$sender", "count": {"$sum": 1}}}
    ])
    unread_dict = {item['_id']: item['count'] for item in unread_counts}
    
    return render_template('Binbox.html', 
                           users=users_list, 
                           unread_dict=unread_dict)

@auth.route('/Fchat/<receiver>')
def Fchat(receiver):
    if 'username' not in session:
        return redirect(url_for('auth.Flogin'))
    messages.update_many(
        {"sender": receiver, "receiver": session['username'], "read": False},
        {"$set": {"read": True}}
    )
    
    chat_history = messages.find({
        "$or": [
            {"sender": session['username'], "receiver": receiver},
            {"sender": receiver, "receiver": session['username']}
        ]
    }).sort("timestamp", 1)
    
    return render_template('Fchat.html', 
                           receiver=receiver, 
                           messages=chat_history)

@auth.route('/Bchat/<receiver>')
def Bchat(receiver):
    if 'username' not in session:
        return redirect(url_for('auth.Blogin'))
    messages.update_many(
        {"sender": receiver, "receiver": session['username'], "read": False},
        {"$set": {"read": True}}
    )

    chat_history = messages.find({
        "$or": [
            {"sender": session['username'], "receiver": receiver},
            {"sender": receiver, "receiver": session['username']}
        ]
    }).sort("timestamp", 1)
    
    return render_template('Bchat.html', 
                           receiver=receiver, 
                           messages=chat_history)

@socketio.on('join')
def on_join(data):
    receiver = data['receiver']
    room = sorted([session['username'], receiver])
    room = '_'.join(room)
    join_room(room)
    send(f"{session['username']} joined the chat", to=room)

@socketio.on('send_message')
def handle_send_message(data):
    if 'username' not in session:
        return
    
    message = {
        "sender": session['username'],
        "receiver": data['receiver'],
        "content": data['content'],
        "timestamp": datetime.utcnow(),
        "read":False
    }
    
    messages.insert_one(message)
    
    room = sorted([session['username'], data['receiver']])
    room = '_'.join(room)
    socketio.emit('receive_message', json_util.dumps(message), room=room)
    


@auth.route('/weath')
def weath():
    return render_template('weath.html')

@auth.route('/profile')
def profile():
    return render_template('profile.html')

@auth.route('/Bprofile')
def Bprofile():
    return render_template('Bprofile.html')


@auth.route('/post', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        caption = request.form['caption']
        image = request.files['image']
        
        if image and allowed_file(image.filename):
            # Generate secure UUID filename
            ext = os.path.splitext(image.filename)[1].lower()
            unique_filename = str(uuid.uuid4()) + ext
            
            # Use absolute path from config
            image_path = os.path.join(
                current_app.config['UPLOAD_FOLDER'],
                secure_filename(unique_filename)
            )
            image.save(image_path)
            
            current_user = Fuser_collection.find_one({'_id': ObjectId(session['user_id'])})
            
            post = {
                'username': current_user['username'],
                'caption': caption,
                'image_filename': unique_filename,
                'timestamp': datetime.utcnow()
            }
            posts.insert_one(post)
            
            return redirect(url_for('auth.Fhome'))
    
    return render_template('post.html')


@auth.route('/Fedit_profile', methods=['GET', 'POST'])
def Fedit_profile():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user = Fuser_collection.find_one({'_id': ObjectId(session['user_id'])})

    if request.method == 'POST':
        new_username = request.form['username']
        new_email = request.form['email']
        new_phonenumber = request.form['phonenumber']

        # Check username availability
        if new_username != user['username']:
            existing = Fuser_collection.find_one({'username': new_username})
            if existing:
                flash('Username already taken')
                return redirect(url_for('auth.Fedit_profile'))

        # Check email availability
        if new_email != user['email']:
            existing = Fuser_collection.find_one({'email': new_email})
            if existing:
                flash('Email already in use')
                return redirect(url_for('auth.Fedit_profile'))
        if new_phonenumber != user['phonenumber']:
            existing = Fuser_collection.find_one({'email': new_phonenumber})
            if existing:
                flash('PhoneNumber already in use')
                return redirect(url_for('auth.Fedit_profile'))

        # Update user data
        Fuser_collection.update_one(
            {'_id': ObjectId(session['user_id'])},
            {'$set': {
                'username': new_username,
                'email': new_email,
                'phonenumber':new_phonenumber
            }}
        )

        # Update session data
        session['username'] = new_username
        session['email'] = new_email
        session['phonenumber'] = new_phonenumber

        flash('Profile updated successfully')
        return redirect(url_for('auth.profile'))

    return render_template('Fedit_profile.html', user=user)

@auth.route('/Bedit_profile', methods=['GET', 'POST'])
def Bedit_profile():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user = Buser_collection.find_one({'_id': ObjectId(session['user_id'])})

    if request.method == 'POST':
        new_username = request.form['username']
        new_email = request.form['email']
        new_phonenumber = request.form['phonenumber']

        # Check username availability
        if new_username != user['username']:
            existing = Buser_collection.find_one({'username': new_username})
            if existing:
                flash('Username already taken')
                return redirect(url_for('auth.Bedit_profile'))

        # Check email availability
        if new_email != user['email']:
            existing = Buser_collection.find_one({'email': new_email})
            if existing:
                flash('Email already in use')
                return redirect(url_for('auth.Fedit_profile'))
        if new_phonenumber != user['phonenumber']:
            existing = Buser_collection.find_one({'email': new_phonenumber})
            if existing:
                flash('PhoneNumber already in use')
                return redirect(url_for('auth.Fedit_profile'))

        # Update user data
        Fuser_collection.update_one(
            {'_id': ObjectId(session['user_id'])},
            {'$set': {
                'username': new_username,
                'email': new_email,
                'phonenumber':new_phonenumber
            }}
        )

        # Update session data
        session['username'] = new_username
        session['email'] = new_email
        session['phonenumber'] = new_phonenumber

        flash('Profile updated successfully')
        return redirect(url_for('auth.Bprofile'))

    return render_template('Bedit_profile.html', user=user)



@auth.route('/AI')
def AI():
    return render_template('AI.html')

@auth.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['AI_FOLDER'], filename)
    file.save(filepath)

    try:
        img = Image.open(filepath).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item() * 100
        
        
        record = {
            'filename': filename,
            'prediction': CLASS_LABELS[predicted_idx],
            'confidence': round(confidence, 2),
            'medicine':medicine,
            'timestamp': datetime.now()
        }
        AI_collection.insert_one(record)

        return jsonify({
            'prediction': CLASS_LABELS[predicted_idx],
            'confidence': f"{confidence:.2f}%",
            'medicine' : f"{medicine}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
