import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : "https://attendance-d9083-default-rtdb.firebaseio.com/"
})

def update_attendance_in_firebase(subject_ref, name):
    try:
        # Retrieve the current attendance value
        student_ref = subject_ref.child(name)
        student_data = student_ref.get()
        if student_data:  # Check if the student exists
            current_attendance = student_data.get("Total_attendance", 0)
            student_ref.update({"Total_attendance": current_attendance + 1})
            print(f"Attendance updated for {name}: {current_attendance + 1}")
        else:
            print(f"No record found for {name} in Firebase.")
    except Exception as e:
        print(f"Error updating attendance: {e}")

sub1 = db.reference("Microprocessor")
sub2 = db.reference("Electromagnetics")
sub3 = db.reference("Object Oriented Programming")

bct_students = { 
    "Mike_Tyson" : 
    {
        "Roll_no" : "KAN078BCT001",
        "Total_attendance" : 0
    },
    
    "Tiger_Woods" : 
    {
        "Roll_no" : "KAN078BCT002",
        "Total_attendance" : 1
    },
    
    "David_Beckham" : 
    {
        "Roll_no" : "KAN078BCT003",
        "Total_attendance" : 7
    },

    "Zinedine_Zidane" : 
    {
        "Roll_no" : "KAN078BCT004",
        "Total_attendance" : 2
    },

    "Roger_Federer" : 
    {
        "Roll_no" : "KAN078BCT0005",
        "Total_attendance" : 1
    },
}

for key,value in bct_students.items():
    sub1.child(key).set(value)
    sub2.child(key).set(value)
    sub3.child(key).set(value)