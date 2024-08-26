import pickle
import streamlit as st

with open('model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Welcome message
def welcome():
    return 'welcome all'


# Prediction function
def prediction(message):
    message_vector = vectorizer.transform([message])
    predict = classifier.predict(message_vector)
    return predict


# Main function for the Streamlit app
def main():
    st.title("Spam Detection")

    # Frontend
    html_temp = '''
    <div style ="background-color:Bisque; padding:13px; border-radius:50px; margin-bottom: 15px">
        <h1 style ="color:black; text-align:center; font-family: Source Sans Pro; font-size: 50px;">Spam Detector</h1> 
    </div>
    <p>
        0 - Not Spam
        <br>
        1 - Spam
    </p>
    '''

    st.markdown(html_temp, unsafe_allow_html=True)

    # Input box for user message
    message = st.text_input("Enter a Message")

    # Predict button
    if st.button("Predict"):
        result = prediction(message)
        st.success('The output is {}'.format(result))


if __name__ == '__main__':
    main()
