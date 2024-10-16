from flask import Flask, request, jsonify
from autoclustering import *
import pandas as pd
import re
from io import StringIO
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import threading

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set up the OpenAI API key using environment variable
api_key = os.getenv("OPENAI_API_KEY")

# # Function to stream data with a delay for a typing effect
# def stream_data(response):
#     for word in response:
#         yield word 
#         time.sleep(0.009)

# # Function to load a PNG image
# def load_image(image_file):
#     return f"./{image_file}"

def get_cluster_stats(df, cluster_column='Cluster'):
    unique_col = [col for col in df.columns if df[col].nunique() == len(df)]
    if len(unique_col) != 0:
        df.set_index(unique_col, inplace=True)

    # Get numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Group by 'Cluster' and calculate mean, max, min for numeric columns
    numeric_cluster_stats = df.groupby(cluster_column)[numeric_columns].agg(['mean', 'max', 'min']).reset_index()

    categorical_columns = df.select_dtypes(include=['object','O']).columns
    # Group by 'Cluster' and calculate mean, max, min for numeric columns
    categorical_cluster_stats = df.groupby(cluster_column)[categorical_columns].agg(['max', 'min']).reset_index()
    return numeric_cluster_stats, categorical_cluster_stats

# Function to interact with OpenAI's GPT model for cluster analysis
def chat_with_clusters(prompt):
    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful marketing data analyst that helps me with customer segmentation!"},
                      {"role": "user", "content": f'''{prompt}'''}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return str(e)

# # Function to parse clusters from the OpenAI result
def extract_clusters(result):
    
    if 'Cluster_:' in result:
        clusters = re.split(r'(Cluster_\d+:)', result)
    else:
        clusters = re.split(r'(cluster_\d+)', result)
    # print(clusters)
    cluster_data = {}
    for i in range(1, len(clusters), 2):
        cluster_no = clusters[i].strip()
        # description = clusters[i + 1].strip()
        cluster_data_points = clusters[i + 1].strip().split('\n')
        cluster_title= cluster_data_points[0].strip()
        cluster_description= cluster_data_points[1].strip()
        key_points = []
        for point in cluster_data_points[2:]:
            key_points.append(point.strip())
        cluster_data[cluster_no] = {
            "title": cluster_title,
            "description": cluster_description,
            "key_points": key_points
        }
        # cluster_data.append((title, description))
    return cluster_data

# @app.route('/audience_segmentation', methods=['POST'])
# def audience_segmentation():
#     try:
#         # Load data from local CSV files
#         data3 = pd.read_csv(request.files['data3'])  # Cluster data
#         data1, data2 = get_cluster_stats(data3)

#         # Replace values in 'Cluster' column with descriptive names
#         data4 = data3

#         # Initialize session state for clusters and result
#         if 'clusters' not in st.session_state:
#             st.session_state.clusters = []

#         if 'result' not in st.session_state:
#             st.session_state.result = ""

#         try:
#             # Create three columns for layout
#             col1, col2, col3 = st.columns([1, 1, 1])

#             with col3:
#                 st.info("Audience Loaded Successfully")
#                 st.dataframe(data1, use_container_width=True)
#                 st.dataframe(data2, use_container_width=True)

#             # User input for chart and analysis
#             with col3:
#                 st.info("Audience Segmentation Visualization")

#                 # Chart options
#                 column_options = data3.columns.tolist()
#                 x_axis = st.selectbox('Select column for X-axis', column_options)
#                 y_axis = st.selectbox('Select column for Y-axis', column_options)
#                 chart_type = st.radio('Select chart type', ('Scatter Plot', 'Bar Chart'))

#                 if x_axis and y_axis:
#                     st.write(f"### {chart_type} of {x_axis} vs {y_axis}")

#                     # Check the data types and prepare for plotting
#                     data4[y_axis] = pd.to_numeric(data4[y_axis], errors='coerce')  # Ensure Y-axis is numeric
#                     x_is_categorical = data4[x_axis].dtype == 'object' or data4[x_axis].dtype.name == 'category'
#                     y_is_numeric = pd.api.types.is_numeric_dtype(data4[y_axis])

#                     plt.figure(figsize=(6, 3))

#                     # Create the appropriate chart based on user selection and data types
#                     if chart_type == 'Scatter Plot':
#                         if not (x_is_categorical and not y_is_numeric):
#                             sns.scatterplot(x=data4[x_axis], y=data4[y_axis])
#                         else:
#                             st.error("Scatter plots require at least one numeric variable.")

#                     elif chart_type == 'Bar Chart':
#                         if x_is_categorical and y_is_numeric:
#                             if data4[y_axis].isnull().all():
#                                 st.error(f"The selected Y-axis ({y_axis}) has no valid numeric values.")
#                             else:
#                                 bar_data = data4.groupby(x_axis)[y_axis].mean().reset_index()  # Calculate mean of the Y-axis by the X-axis categories
#                                 sns.barplot(x=bar_data[x_axis], y=bar_data[y_axis])
#                         else:
#                             st.error("Bar charts require the X-axis to be categorical and the Y-axis to be numeric.")

#                     plt.title(f"{chart_type} of {x_axis} vs {y_axis}")
#                     st.pyplot(plt)

#             st.dataframe(data4, use_container_width=True)

#             with col1:
#                 # Button to summarize audience segments
#                 if st.button("Summarize Audience Segments"):
#                     with st.spinner("Running Autocluster..."):
#                         autocluster('../../data.csv')

#                     with st.spinner("Summarizing..."):
#                         prompt = f'''Here are the clusters we are going to analyze. There are two datasets: one for numerical variables is {data1} and one for 
# categoricals is {data2}. Followed by a brief description of each cluster and pithy, descriptive persona titles for each you would associate with each cluster. 
#                         Justify the cluster persona names with bullet point statistics.
#                         Use format
#                         ''old_cluster_name: The new name
#                         debrief description of this cluster
#                             1.key point_1
#                             2.key point_2
#                             3.key point_3
#                             4.key point_4
#                             n.key point_n

#                         old_cluster_name: The new name
#                         debrief description of this cluster
#                             1.key point_1
#                             2.key point_2
#                             3.key point_3
#                             4.key point_4
#                             n.key point_n

#                         old_cluster_name: The new name 
#                         debrief description of this cluster
#                             1.key point_1
#                             2.key point_2
#                             3.key point_3
#                             4.key point_4
#                             n.key point_n''
#                         the output should be only like the given format
#                         '''

#                     result = chat_with_clusters(prompt)

#                     # Store result in session state
#                     st.session_state.result = result

#                     # Parse the result into clusters and store in session state
#                     st.session_state.clusters = extract_clusters(result)
                

#             # Display clusters in a collapsible dropdown in Column 1
#                 count = 0  # Start count at 0 for segment_0

#                 # print(st.session_state.result)
                
#                 # print(st.session_state.clusters)
#                 for title, description in st.session_state.clusters:
#                     print(title)
#                     with st.expander(f"Segment_{count}", expanded=False):
#                         st.text_area(label=f"Segment_{count}", value=description, height=200, key=f"cluster_text_area_{count}",label_visibility="hidden")
#                         count += 1  # Increment count for each segment

#                 # Prepare the result for download
#                 if st.session_state.result:
#                     buffer = StringIO()
#                     buffer.write(st.session_state.result)
#                     buffer.seek(0)

#                     st.download_button(
#                         label="Download Summary",
#                         data=buffer.getvalue(),
#                         file_name="audience_summary.txt",
#                         mime="text/plain"
#                     )

#             with col2:
#                 # Create a container for the chat interface
#                 chat_container = st.container(height=400, border=True)
#                 # Load SVG images
#                 audience_ai_avatar_svg = load_image("images/AudienceMatchAI avatar 5.png")

#                 with chat_container:
#                     st.markdown(
#                         """
#                         <style>
#                         .stContainer {
#                             background-color: #FFA500;
#                         }
#                         </style>
#                         """,
#                         unsafe_allow_html=True
#                     )   
                
#                 # Initialize chat history if it doesn't exist
#                 if "messages" not in st.session_state or len(st.session_state.messages) ==0 :
#                     with chat_container:
#                         with st.spinner("Loading..."):
#                             time.sleep(2)
#                         st.chat_message("AudeinceAI", avatar=audience_ai_avatar_svg).write_stream(stream_data("AudienceAI: HI 👋, I am your AI assistant, AudeinceAI. How can I assist you with the audience segments, based on the Audience Segmentation Summary?"))
#                         st.chat_message("AudeinceAI", avatar=audience_ai_avatar_svg).write_stream(stream_data("AudienceAI: Please get the Audeince Segments First"))
#                     st.session_state.messages = []
                               
#                 # Display chat messages from history on app rerun
#                 with chat_container:
#                     for message in st.session_state.messages:
#                         # for aliging the user message on rigth and audience AI message on left
#                         if message['role'] == 'User':
#                             with st.chat_message(message['role'], avatar=message.get('avatar')):
#                                 st.markdown(f"<div style='text-align: right;'>{message['content']}</div>", unsafe_allow_html=True)
#                     # Right align user messages
                       
#                     elif message['role'] == 'AudeinceAI':
#                         with st.chat_message(message["role"], avatar=message.get("avatar")):
#                             st.write(message["content"])
#                     #     with st.chat_message(message['role'], avatar=message.get('avatar')):
#                     #         st.markdown(f"<div style='text-align: left;'>{message['content']}</div>", unsafe_allow_html=True)
               
#                             # with st.chat_message(message["role"], avatar=message.get("avatar")):
#                     #     st.write(message["content"])
        
#                 # Handle user input and generate AI response
#                 if prompt := st.chat_input("Ask your query for audience segments Here"):
#                     # Add user message to chat history
#                     st.session_state.messages.append({"role": "User", "content": prompt, "avatar": "👨‍💼"})
#                     with chat_container.chat_message("User", avatar="👨‍💼"):
#                         st.markdown(f"<div style='text-align: right;'>{prompt}</div>", unsafe_allow_html=True)
                   

#                 # Generate and display AI response if the last message is not from AI
#                 if st.session_state.messages and st.session_state.messages[-1]["role"] != "AudeinceAI":  
#                     with chat_container:
#                         with st.chat_message("AudeinceAI", avatar=audience_ai_avatar_svg):
#                             with st.spinner("Thinking..."):      
#                                 # Construct the full conversation history for the prompt
#                                 if st.session_state.result:  # Only include result if it exists
#                                     data1_string = data1.to_string(index=False)  # Convert data1 to string without the index
#                                     data2_string = data2.to_string(index=false)  # Convert data2 to string without the index
#                                     initial_prompt = (
#                                         f"This is the cluster information you should only answer questions based on this: {st.session_state.result}\n"
#                                         f"Here are the numerical data:\n{data1_string}\n"
#                                         f"Here are the categorical data:\n{data2_string}"
#                                     )
#                                 else:
#                                     initial_prompt = "This is the cluster information you should only answer questions based on this: No cluster information available."

#                                 full_conversation = "\n\n".join(f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages)
#                                 chat_input = f'''{initial_prompt}\n{full_conversation}'''
#                                 response = chat_with_clusters(chat_input)
#                             st.write_stream(stream_data(response))
#                         # Add assistant response to chat history
#                         st.session_state.messages.append({"role": "AudeinceAI", "content": response, "avatar": audience_ai_avatar_svg})

#                         # Force a rerun to update the chat display
#                         # st.rerun()

#     except Exception as e:
#         return str(e)



@app.route('/', methods=['GET'])
def home():
    file_path = r"model_output/clustering_results.csv"
    data_path = r"input_data/data.csv"
    if os.path.exists(file_path):
        return "Welcome to the home page"
    else:
        autocluster(data_path)
        return "File not found, running autocluster"
    

@app.route('/cluster_summary', methods=['GET'])
def cluster_summarizor():
    file_path = r"model_output/clustering_results.csv"
    cluster_data = pd.read_csv(file_path)
    print("creating Numeric and Categorical stats Data")
    data1, data2 = get_cluster_stats(cluster_data)
    print("creating Clusters Summary")
    prompt = f'''Here are the clusters we are going to analyze. There are two datasets: one for numerical variables is {data1} and one for 
categoricals is {data2}. Followed by a brief description of each cluster and pithy, descriptive persona titles for each you would associate with each cluster. 
                        Justify the cluster persona names with bullet point statistics.
                        Use format
                        ''old_cluster_name: The new name
                        debrief description of this cluster
                            1.key point_1
                            2.key point_2
                            3.key point_3
                            4.key point_4
                            n.key point_n

                        old_cluster_name: The new name
                        debrief description of this cluster
                            1.key point_1
                            2.key point_2
                            3.key point_3
                            4.key point_4
                            n.key point_n

                        old_cluster_name: The new name 
                        debrief description of this cluster
                            1.key point_1
                            2.key point_2
                            3.key point_3
                            4.key point_4
                            n.key point_n''
                        the output should be only like the given format
                        please make sure always the old_cluster_name starts with Cluster_
                        '''

    result = chat_with_clusters(prompt)
    print(result)
    clusters_summary_json  = extract_clusters(result)
    # print(clusters_summary_json)
    result_json = jsonify({"json_output": clusters_summary_json, 
                           "string_output": result,
                           "numeric_data": data1.to_html(classes="table table-striped"),
                           "categorical_data": data2.to_html(classes="table table-striped")})
    return result_json



     
@app.route('/chat', methods=['POST'])
def chat_with_audiences():
    data = request.get_json()
    history = data.get('history', '')
    user_query = data.get('query', '')
    conversation_history = data.get('conversation_history', '')
    print(history)
    print('*'*100)
    print(user_query)
    print(conversation_history)
    if history:  # Only include result if it exists
        file_path = r"model_output/clustering_results.csv"
        cluster_data = pd.read_csv(file_path)
        print("creating Numeric and Categorical stats Data")
        data1, data2 = get_cluster_stats(cluster_data)
        data1_string = data1.to_string(index=False)  # Convert data1 to string without the index
        data2_string = data2.to_string(index=False)  # Convert data2 to string without the index
        initial_prompt = (
            f"This is the cluster information you should only answer questions based on this: {history}\n"
            f"Here are the numerical data:\n{data1_string}\n"
            f"Here are the categorical data:\n{data2_string}"
        )
    else:
        initial_prompt = "This is the cluster information you should only answer questions based on this: No cluster information available."

    conversation_history = conversation_history + f"\n\nUser: {user_query}"
    full_conversation = f"{initial_prompt}\n{conversation_history}"
    response = chat_with_clusters(full_conversation)
    conversation_history = conversation_history + f"\nAudienceAI: {response}"
   
    return jsonify({"response": response,
                    "conversation_history": conversation_history,
                    "history": history})

    # return response

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8084)