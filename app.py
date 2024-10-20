import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import pipeline
import re

# Streamlit app layout with custom CSS styles
st.markdown(
    """
    <style>
    .summary-section, .analysis-section, .comparison-section {
        background-color: #e4e4f5;
        padding-left:15px;
        border-radius: 10px;
        display: flex;
        align-items: center;
    }
    h3 {
        
        margin-top: 0;
        color: darkblue;  /* Set font color to black */
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Streamlit app title and input
st.title("Multi-Model Text Summarization with Vectorization Insights")
st.text("This app shows comparison of summaries using different models wrt. vectorization.")

# Input box for user text with larger box for copy-paste functionality
user_text = st.text_area("Paste your text here:", height=200)

# Function to highlight important words
def highlight_summary(summary, top_words_count, top_words_tfidf):
    summary_lower = summary.lower()
    st.markdown(
        """
        <style>
        mark {
            background-color: #ffd54f;
            color: black;
            padding: 2px 4px;
            border-radius: 4px;
            font-weight: bold;
            box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    for word in set(top_words_count + top_words_tfidf):
        summary_lower = summary_lower.replace(word, f"<mark><strong>{word}</strong></mark>")
    highlighted_summary = ""
    summary_words = summary.split()
    summary_lower_words = summary_lower.split()
    for original, lower in zip(summary_words, summary_lower_words):
        if "<mark>" in lower:
            highlighted_summary += f" {lower}"
        else:
            highlighted_summary += f" {original}"
    return highlighted_summary.strip()

# Function to generate summaries
def generate_summaries(input_text):
    t5_summarizer = pipeline("summarization", model="t5-small")
    pegasus_summarizer = pipeline("summarization", model="google/pegasus-xsum")
    distilbart_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    t5_summary = t5_summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    pegasus_summary = pegasus_summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    distilbart_summary = distilbart_summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    return t5_summary, pegasus_summary, distilbart_summary

# Vectorization and analysis
def analyze_text(input_text):
    count_vectorizer = CountVectorizer(stop_words='english')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform([input_text])
    tfidf_matrix = tfidf_vectorizer.fit_transform([input_text])
    count_vocab = count_vectorizer.get_feature_names_out()
    tfidf_vocab = tfidf_vectorizer.get_feature_names_out()
    word_counts = count_matrix.toarray().flatten()
    word_tfidf = tfidf_matrix.toarray().flatten()
    count_dict = dict(zip(count_vocab, word_counts))
    tfidf_dict = dict(zip(tfidf_vocab, word_tfidf))
    sorted_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    top_words_count = [word.lower() for word, _ in sorted_count]
    top_words_tfidf = [word.lower() for word, _ in sorted_tfidf]
    return top_words_count, top_words_tfidf, sorted_count, sorted_tfidf, count_vocab

# Function to count highlighted words in summaries
def count_highlighted_words_in_summary(summary, top_words_count, top_words_tfidf):
    summary_words = re.findall(r'\b\w+\b', summary.lower())
    count_top_words = len([word for word in summary_words if word in top_words_count])
    tfidf_top_words = len([word for word in summary_words if word in top_words_tfidf])
    return count_top_words, tfidf_top_words

# Visualization function
# Plot graphs using Plotly
def plot_graphs(sorted_count, sorted_tfidf):
    count_words, count_values = zip(*sorted_count)
    tfidf_words, tfidf_values = zip(*sorted_tfidf)

    # CountVectorizer plot
    fig1 = go.Figure([go.Bar(x=count_values, y=count_words, orientation='h', marker=dict(color='dodgerblue'))])
    fig1.update_layout(title="Top 10 Words by CountVectorizer", xaxis_title='Word Count', yaxis_title='Words')
    st.plotly_chart(fig1, use_container_width=True)

    # TF-IDF plot
    fig2 = go.Figure([go.Bar(x=tfidf_values, y=tfidf_words, orientation='h', marker=dict(color='mediumseagreen'))])
    fig2.update_layout(title="Top 10 Words by TfidfVectorizer", xaxis_title='TF-IDF Score', yaxis_title='Words')
    st.plotly_chart(fig2, use_container_width=True)

if st.button("Analyze Text and Generate Summaries") and user_text.strip():
    # Perform analysis and generate summaries
    top_words_count, top_words_tfidf, sorted_count, sorted_tfidf, count_vocab = analyze_text(user_text)
    t5_summary, pegasus_summary, distilbart_summary = generate_summaries(user_text)

    # Section: Display highlighted summaries
    st.markdown('<div class="summary-section"><h3>T5 Summary</h3>', unsafe_allow_html=True)
    st.markdown(highlight_summary(t5_summary, top_words_count, top_words_tfidf), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="summary-section"><h3>Pegasus Summary</h3>', unsafe_allow_html=True)
    st.markdown(highlight_summary(pegasus_summary, top_words_count, top_words_tfidf), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="summary-section"><h3>DistilBART Summary</h3>', unsafe_allow_html=True)
    st.markdown(highlight_summary(distilbart_summary, top_words_count, top_words_tfidf), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section: Detailed Text Analysis
    st.markdown('<div class="analysis-section"><h3>Detailed Text Analysis</h3>', unsafe_allow_html=True)
    for summary_text, model_name in zip([t5_summary, pegasus_summary, distilbart_summary], 
                                        ["T5", "Pegasus", "DistilBART"]):
        st.markdown(f"#### {model_name} Summary Analysis")
        col1, col2 = st.columns(2)
        with col1:
            total_words_original = len(re.findall(r'\w+', user_text))
            total_words_summary = len(re.findall(r'\w+', summary_text))
            st.write(f"**Total words in the text:** {total_words_original}")
            st.write(f"**Words in summary:** {total_words_summary}")
        with col2:
            stop_words_removed = total_words_original - len(count_vocab)
            unique_words = len(count_vocab)
            st.write(f"**Stop words removed:** {stop_words_removed}")
            st.write(f"**No. of words after stop words removed:** {unique_words}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Section: Comparison of Highlighted Words
    st.markdown('<div class="comparison-section"><h3>Comparison of Highlighted Words in Summaries</h3>', unsafe_allow_html=True)
    st.write("")
    for summary, name in zip([t5_summary, pegasus_summary, distilbart_summary], 
                             ["T5", "Pegasus", "DistilBART"]):
        count_top, tfidf_top = count_highlighted_words_in_summary(summary, top_words_count, top_words_tfidf)
        st.write(f"**{name} Summary:** {count_top}/10 Top words from Vectorization")
    st.markdown('</div>', unsafe_allow_html=True)

    # Section: Word Analysis and Plot
    st.markdown('<div class="analysis-section"><h3>Vectorization-Based Word Analysis</h3>', unsafe_allow_html=True)
    st.write("")
    plot_graphs(sorted_count, sorted_tfidf)
    st.markdown('</div>', unsafe_allow_html=True)