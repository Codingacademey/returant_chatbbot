import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Grand Avenue Restaurant Chatbot",
    page_icon="🍽️",
    layout="wide"
)

# Restaurant Info
restaurant_name = "Grand Avenue Restaurant"
location = "Khajuriwal, Head marala road, Sialkot"
contact = "03046001463 | 052533000"
timings = "Monday-Sunday | 12:00 PM - 12:00 AM"

# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'menu_view' not in st.session_state:
    st.session_state.menu_view = False

# Load and process PDF data
loader = PyPDFLoader("data.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0,max_tokens=None,timeout=None)

system_prompt = (
   """
   You are a friendly and professional restaurant chatbot named "Grand Avenue Assistant" for the restaurant "Restaurant i Events | OutdoorPark" located in Sialkot.



You assist customers with:
1. 📋 **Menu Inquiry**
2. 🪑 **Table Booking / Order Placement**
3. 🕰 **Opening & Closing Hours**
4. 📍 **Location & Contact**
5. ❓ **FAQs**
6. 🤝 **Follow-up Questions to keep the conversation engaging**

---

### 🎯 Response Guidelines:
-> when user ask about menu ,  you tell  all category of menus.


#### 🧾 MENU
- if user ask about any kind of facilities: give them the ans like we provide the facilities of outdoor , indoor servises , table reservation, taste our special items etc.
If the user asks for the menu or mentions an item (like BBQ, pizza, shakes, etc.):
- Respond with item names and prices.
- Add a follow-up like:  
  "Would you like to see more items from our [category] menu?"  
  OR  
  "Would you like to place an order for this item?"

#### 🪑 BOOKING / ORDER
If the user says:
> "book a table", "make a reservation", "place an order", or "order food"

- Respond:  
  "Sure! 🎉 You can book a table or place your order by filling this quick form:  
  👉 [Booking & Order Form](https://docs.google.com/forms/d/e/1FAIpQLScDthlNGEvIDWap3qVmmHt4jg5XEDgQuQpHdkjr6sQ3UwwdRw/viewform)"
- Follow-up with:  
  "Would you like help exploring the menu while you book?"

  for location this is google map link : 

#### ⏰ TIMINGS
If asked about opening/closing times:
- Respond:  
  "⏰ We're open every day from 12:00 PM to 12:00 AM!"
- Follow-up:  
  "Would you like to know our busiest hours or the best time to visit?"

#### 📍 LOCATION / CONTACT
If asked about where you are or how to call:
- Respond:  
  "📍 We're located at Khajuriwal, Head Marala Road, Sialkot.  
  📞 Contact us at: 03046001463 or 052533000"
- Follow-up:  
  "Want directions or a Google Maps link?"

#### ❓ FAQs / OTHERS
If asked about delivery, payment, birthday deals, etc.:
- Respond with whatever info is in the dataset.
- If not available, say:  
  "Let me check that for you! In the meantime, would you like to view our specials or reserve a table?"

---

### 🔁 Conversational Flow Tips:
- Always end responses with a soft **follow-up** question.
- Stay friendly, professional, and responsive.
- Break down long replies with spacing and emojis.

---
{context}
"""
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# Create memory instance
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

# Create the conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    return_generated_question=True,
)

# --- Menu Data (copied from RestaurantRobot/menu_data.py) ---
menu_categories = {
    "Shakes": [
        {"name": "Mango Shake", "price": 350},
        {"name": "Banana Shake", "price": 300},
        {"name": "Strawberry Shake", "price": 450},
        {"name": "Power Shake", "price": 500}
    ],
    "Special Drinks": [
        {"name": "Mint Margarita", "price": 300},
        {"name": "Lemonade", "price": 280},
        {"name": "Pina Colada", "price": 350},
        {"name": "Sweet Lassi", "price": 300},
        {"name": "Saltish Lassi", "price": 300}
    ],
    "Ice Cream Shake": [
        {"name": "Chocolate Shake", "price": 450},
        {"name": "Strawberry Shake", "price": 400},
        {"name": "Oreo Shake", "price": 400},
        {"name": "Vanilla Shake", "price": 400}
    ],
    "Soft Drink": [
        {"name": "Pepsi Cane", "price": 120},
        {"name": "Coke Cane", "price": 120},
        {"name": "Fresh Lime 7Up", "price": 150},
        {"name": "Fresh Lime Soda", "price": 150},
        {"name": "Coke 1.5 Liter Drink", "price": 220},
        {"name": "Pepsi 1.5 Liter Drink", "price": 220},
        {"name": "Mineral Water", "price": 120}
    ],
    "Coffee & Tea": [
        {"name": "Coffee", "price": 350},
        {"name": "Chaye Doodh Patti", "price": 200}
    ],
    "Wings & Appetizers": [
        {"name": "Hot Wings", "price": 800},
        {"name": "Grill Wings", "price": 800},
        {"name": "Honey Wings", "price": 800},
        {"name": "Chicken Nuggets", "price": 700},
        {"name": "Fried Fish", "price": 1700},
        {"name": "Loaded Fries", "price": 599},
        {"name": "Plain Fries", "price": 400}
    ],
    "Sandwich": [
        {"name": "Chicken Grill Sandwich", "price": 799},
        {"name": "Chicken Club Sandwich", "price": 799}
    ],
    "Tandoor Section": [
        {"name": "Roghni Naan", "price": 150},
        {"name": "Garlic Naan", "price": 160},
        {"name": "Kalwanji Naan", "price": 150},
        {"name": "Kashmiri Naan", "price": 400},
        {"name": "Cheese Naan", "price": 350},
        {"name": "Tandoori Paratha", "price": 150},
        {"name": "Chicken White Spicy Naan", "price": 350},
        {"name": "Desi Roti", "price": 40},
        {"name": "White Roti", "price": 50},
        {"name": "Roti Per Head", "price": 120}
    ],
    "Sweets": [
        {"name": "Special Shahi Kheer", "price": 799},
        {"name": "Garam Gulab Jamun", "price": 799},
        {"name": "Gajjar Halwa", "price": 1200},
        {"name": "Ice Cream (2 Scoops)", "price": 300}
    ],
    "Special Platters": [
        {
            "name": "Norangi Platter (2 Person)",
            "price": 3900,
            "description": "2 Piece Malai Botti, 2 Piece Tikka Botti, 2 Piece Kalmi Tikka, 2 Piece Chicken Kabab, 2 Piece Beef Kabab, Half Chicken Fried Rice"
        },
        {
            "name": "Grand Avenue Special Family Platter (8 Person)",
            "price": 12999,
            "description": "1 Seekh Tikka Botti, 1 Seekh Malai Botti, 1 Seekh Haryali Botti, 1 Seekh Kalmi Tikka, 4 Piece Reshmi Kabab, 4 Piece Cheese Kabab, 4 Piece Beef Kabab, 1 Seekh Mutton Chanp, 1 Kashmiri Pulao, 1 Chicken Afghani Karahi, Rotti – Naan – Raita"
        },
        {
            "name": "Bar. B. Q. Shahjahan Platter (6 Person)",
            "price": 8999,
            "description": "1 Seekh Haryali Botti, 1 Seekh Pasha Botti, 1 Seekh Malai Botti, 2 Mutton Chanp, 2 Piece Chicken Kabab, 2 Piece Chicken Reshmi Kabab, Half Chicken Handi, Half Chicken Biryani, 14 Rotti, 2 Naan"
        }
    ],
    "Pizza": [
        {"name": "Tikka Pizza (S)", "price": 900},
        {"name": "Tikka Pizza (M)", "price": 1199},
        {"name": "Tikka Pizza (L)", "price": 1499},
        {"name": "Fajita Pizza (S)", "price": 900},
        {"name": "Fajita Pizza (M)", "price": 1199},
        {"name": "Fajita Pizza (L)", "price": 1499},
        {"name": "Cheese Stuff Pizza (S)", "price": 900},
        {"name": "Cheese Stuff Pizza (M)", "price": 1199},
        {"name": "Cheese Stuff Pizza (L)", "price": 1499},
        {"name": "Crown Stuff Pizza (S)", "price": 1000},
        {"name": "Crown Stuff Pizza (M)", "price": 1300},
        {"name": "Crown Stuff Pizza (L)", "price": 1600},
        {"name": "Hot Spicy Pizza (S)", "price": 1000},
        {"name": "Hot Spicy Pizza (M)", "price": 1300},
        {"name": "Hot Spicy Pizza (L)", "price": 1600},
        {"name": "Super Supreme Pizza (S)", "price": 1000},
        {"name": "Super Supreme Pizza (M)", "price": 1300},
        {"name": "Super Supreme Pizza (L)", "price": 1600},
        {"name": "Malai Botti Pizza (S)", "price": 1000},
        {"name": "Malai Botti Pizza (M)", "price": 1300},
        {"name": "Malai Botti Pizza (L)", "price": 1600},
        {"name": "Special Grand Avenue Pizza (S)", "price": 1199},
        {"name": "Special Grand Avenue Pizza (M)", "price": 1499},
        {"name": "Special Grand Avenue Pizza (L)", "price": 1899},
        {"name": "Lazania Pizza (S)", "price": 1000},
        {"name": "Lazania Pizza (M)", "price": 1300},
        {"name": "Lazania Pizza (L)", "price": 1600},
        {"name": "Donner Pizza (S)", "price": 1400},
        {"name": "Donner Pizza (M)", "price": 1700},
        {"name": "Donner Pizza (L)", "price": 2100},
        {"name": "Cheese Stick", "price": 600}
    ],
    "Burgers": [
        {"name": "Pizza Burger", "price": 699},
        {"name": "Zinger Burger", "price": 499},
        {"name": "Chicken Patty Burger", "price": 500},
        {"name": "Chicken Grill Burger", "price": 699},
        {"name": "Bar. B. Q. Supreme Burger", "price": 600},
        {"name": "Double Dacker Burger", "price": 999},
        {"name": "Tower Burger", "price": 799},
        {"name": "Tender Burger", "price": 599}
    ],
    "Wraps": [
        {"name": "Chicken Nuggets Wrap", "price": 999},
        {"name": "Chicken Arabic Wrap", "price": 800},
        {"name": "Chicken Grill Wrap", "price": 700},
        {"name": "Chicken Zinger Special Wrap", "price": 650},
        {"name": "Shahi Malai Wrap", "price": 750}
    ],
    "Pakistani Dishes": [
        {"name": "Shahi Daal Mash", "price": 499},
        {"name": "Mix Vegetable", "price": 499}
    ],
    "Bar. B. Q.": [
        {"name": "Angara Chicken", "price": 1999},
        {"name": "Chicken Tikka Piece", "price": 390},
        {"name": "Chicken Tikka Botti (10pc)", "price": 899},
        {"name": "Chicken Malai Botti (10pc)", "price": 1299},
        {"name": "Chicken Haryali Botti (10pc)", "price": 1199},
        {"name": "Pasha Botti (10pc)", "price": 1199},
        {"name": "Chicken Darbari Botti (10pc)", "price": 1199},
        {"name": "Kalmi Tikka (6pc)", "price": 1199},
        {"name": "Mutton Chanp (6pc)", "price": 1999},
        {"name": "Fish Tikka (10pc)", "price": 1899},
        {"name": "Chicken Seekh Kabab (4pc)", "price": 1199},
        {"name": "Chicken Reshmi Kabab (4pc)", "price": 1299},
        {"name": "Chicken Cheese Kabab (4pc)", "price": 1399},
        {"name": "Beef Kabab (4pc)", "price": 1399},
        {"name": "Grill Fish", "price": 1450}
    ],
    "Chinese Rice": [
        {"name": "Special Grand Avenue Rice", "price": 900},
        {"name": "Chicken Fried Rice", "price": 900},
        {"name": "Chicken Masala Rice", "price": 900},
        {"name": "Vegetable Rice", "price": 700},
        {"name": "Egg Fried Rice", "price": 800},
        {"name": "Plain Rice", "price": 400}
    ],
    "Thai Foods": [
        {"name": "Chicken Cashewnut", "price": 1400},
        {"name": "Baszil Beef", "price": 1400},
        {"name": "Beef Chilli Dry", "price": 1599},
        {"name": "Handry Beef", "price": 1599}
    ],
    "Continental": [
        {"name": "Chicken American Steak", "price": 1450},
        {"name": "Chicken Pizza Steak", "price": 1250},
        {"name": "Chicken Butter Steak", "price": 1250},
        {"name": "Chicken Taragon Steak", "price": 1350},
        {"name": "Beef American Steak", "price": 1990},
        {"name": "Beef Pizza Steak", "price": 1850},
        {"name": "Beef Butter Steak", "price": 1850},
        {"name": "Beef Taragon Steak", "price": 1850},
        {"name": "Chicken Shashlik With Rice", "price": 1200}
    ],
    "Boneless Cuisine (Pakistani)": [
        {"name": "Special Grand Avenue Mughlai Handi", "price": 1899},
        {"name": "Chicken Handi", "price": 1799},
        {"name": "Chicken Achari Handi", "price": 1800},
        {"name": "Chicken Cheese Handi", "price": 1990},
        {"name": "Chicken Hari Mirch", "price": 1499},
        {"name": "Chicken Green Chilli Lime", "price": 1499},
        {"name": "Chicken Jalfrezi", "price": 1499},
        {"name": "Chicken Ginjer", "price": 1499}
    ],
    "Chicken with Bone (Pakistani)": [
        {"name": "Special Grand Avenue Chicken Afghani (Half)", "price": 1150},
        {"name": "Special Grand Avenue Chicken Afghani (Full)", "price": 1900},
        {"name": "Chicken Lahori Karahi (Half)", "price": 1000},
        {"name": "Chicken Lahori Karahi (Full)", "price": 1800},
        {"name": "Chicken Achari Karahi (Half)", "price": 1000},
        {"name": "Chicken Achari Karahi (Full)", "price": 1800},
        {"name": "Chicken White Karahi (Half)", "price": 1000},
        {"name": "Chicken White Karahi (Full)", "price": 1850}
    ],
    "Mutton with Bone (Pakistani)": [
        {"name": "Special Grand Avenue Mutton Afghani (Half)", "price": 2050},
        {"name": "Special Grand Avenue Mutton Afghani (Full)", "price": 3950},
        {"name": "Mutton Lahori Karahi (Half)", "price": 2000},
        {"name": "Mutton Lahori Karahi (Full)", "price": 3800},
        {"name": "Mutton Achari Karahi (Half)", "price": 2000},
        {"name": "Mutton Achari Karahi (Full)", "price": 3800},
        {"name": "Mutton White Karahi (Half)", "price": 2000},
        {"name": "Mutton White Karahi (Full)", "price": 3850}
    ],
    "Salad Bar": [
        {"name": "Italian Salad", "price": 900},
        {"name": "Russian Salad", "price": 900},
        {"name": "Fruit Salad", "price": 1000},
        {"name": "Kachumber Salad", "price": 499},
        {"name": "Fresh Salad", "price": 399},
        {"name": "Mint Raita", "price": 200},
        {"name": "Zeera Raita", "price": 180},
        {"name": "Plump Sauce", "price": 250}
    ],
    "Starter": [
        {"name": "Garlic Naan", "price": 150},
        {"name": "Cheese Naan", "price": 350},
        {"name": "Dahka Chicken", "price": 1100},
        {"name": "Drum Stick (4pc)", "price": 1499},
        {"name": "Fish Cracker", "price": 450}
    ],
    "Chinese Soup": [
        {"name": "Special Grand Avenue Soup", "price": 999},
        {"name": "Hot & Sour Soup", "price": 899},
        {"name": "Chicken Corn Soup", "price": 799},
        {"name": "Chicken Thai Soup", "price": 899},
        {"name": "Clear Vegetable Soup", "price": 700}
    ],
    "Chinese Gravy": [
        {"name": "Special Grand Avenue Chicken Mangolian", "price": 1199},
        {"name": "Kung Pao Chicken", "price": 1099},
        {"name": "Almond Chicken", "price": 1099},
        {"name": "Dragon Chicken", "price": 1099},
        {"name": "Samsi Chilli Chicken", "price": 1200},
        {"name": "Bars Chicken Hot Spicy", "price": 1100},
        {"name": "Chicken Chilli Dry", "price": 1299},
        {"name": "Chicken Manchurian", "price": 1000},
        {"name": "Chicken Black Pepper", "price": 1099}
    ],
    "Noodles": [
        {"name": "Special Grand Avenue Chicken Chowmein", "price": 1500},
        {"name": "Chicken Chowmein", "price": 1300},
        {"name": "Vegetable Chowmein", "price": 1100},
        {"name": "American Chop Suey", "price": 1100}
    ]
}

# --- Display Menu Section (copied from RestaurantRobot/utils.py) ---
def display_menu_section(category):
    st.subheader(category)
    items = menu_categories[category]
    has_description = any('description' in item for item in items)
    if has_description:
        for item in items:
            with st.expander(f"{item['name']} - Rs. {item['price']}"):
                st.write(item.get('description', ''))
    else:
        items_df = pd.DataFrame(items)
        if 'description' in items_df.columns:
            items_df = items_df.drop(columns=['description'])
        items_df['price'] = items_df['price'].apply(lambda x: f"Rs. {x}")
        items_df = items_df.rename(columns={'name': 'Item', 'price': 'Price'})
        st.table(items_df)

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.image("image.jpeg", caption="Welcome to Grand Avenue Restaurant")
    
    st.header("Restaurant Information")
    st.write(f"**Location:** {location}")
    st.write(f"**Contact:** {contact}")
    st.write(f"**Timings:** {timings}")
    
    # Display a featured platter image
    st.image("image2.jpg", caption="Our Special Platters")

with col2:
    # Toggle between chat and menu view
    menu_tab, chat_tab = st.tabs(["Menu", "Chat with us"])
    
    with menu_tab:
        st.header(f"{restaurant_name} Menu")
        st.image("demo.jpg", 
                 caption="Our Menu")
        
        # Menu categories section
        category_selector = st.selectbox("Select a category to view menu items:", options=list(menu_categories.keys()))
        if category_selector:
            display_menu_section(category_selector)
    
    with chat_tab:
        st.image("demo.jpg", use_column_width="auto")
        
        st.header("Chat with our Restaurant Assistant")
        st.write("Ask me about our menu, booking a table, or placing an order!")
        
        # Display chat history (no custom CSS)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input (appears above conversation blocks)
        query = st.chat_input("Type your question here...")

        if query:
            # Display user message
            with st.chat_message("user"):
                st.write(query)
            
            # Get response from chain
            response = qa_chain({"question": query})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response["answer"])
                if "book a table" in query.lower():
                    st.markdown("👉 [Book a Table Here](https://docs.google.com/forms/d/e/1FAIpQLScDthlNGEvIDWap3qVmmHt4jg5XEDgQuQpHdkjr6sQ3UwwdRw/viewform)")
                if "order" in query.lower() and "place" in query.lower():
                    st.markdown("👉 [Place Your Order Here](https://docs.google.com/forms/d/e/1FAIpQLScDthlNGEvIDWap3qVmmHt4jg5XEDgQuQpHdkjr6sQ3UwwdRw/viewform)")
            
            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

# Footer
st.markdown("---")
st.caption(f"© 2024 {restaurant_name}. All rights reserved.")
