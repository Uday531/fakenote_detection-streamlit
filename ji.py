from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Operating Systems: Files & Processes', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

    def chapter_code(self, code):
        self.set_font('Courier', '', 10)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 5, code, 0, 'L', True)
        self.ln()

pdf = PDF()
pdf.add_page()

# --- CONTENT START ---

# Part 1
pdf.chapter_title('Part 1: Files and File Systems')

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '1. File Concept & Structure', 0, 1)
pdf.chapter_body('A file is a named collection of related information recorded on secondary storage. The OS uses a layered approach to hide hardware complexity.')
pdf.chapter_code("""[ Application Programs ]
       |
[ Logical File System ] (Metadata/Directory structure)
       |
[ File-Organization Module ] (Maps logical blocks to physical)
       |
[ Basic File System ] (Generic driver commands)
       |
[ I/O Control ] (Device Drivers)""")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '2. I-nodes (Index Nodes)', 0, 1)
pdf.chapter_body('An Inode is a data structure in Unix/Linux describing a file-system object. It contains all metadata (permissions, owner, size, timestamps) except the file name.')
pdf.chapter_code("""+------------------+
| Mode / Perms     |
| Owner ID / Group |
| File Size        |
| Timestamps       |
+------------------+
| Direct Block 0   | ----> [ Data ]
| ...              |
| Indirect Block   | ----> [ Pointer Block ] -> [ Data ]
+------------------+""")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '3. Library Functions vs System Calls', 0, 1)
pdf.chapter_body('Standard I/O (Library): High-level (stdio.h), buffered, uses "FILE *". Examples: fopen, fprintf.\nSystem Calls (Kernel): Low-level (unistd.h), unbuffered, uses File Descriptors (int). Examples: open, read, write.')

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '4. Kernel Support (The 3 Tables)', 0, 1)
pdf.chapter_body('The kernel tracks open files using three linked tables, allowing processes to share files while maintaining independent offsets.')
pdf.chapter_code("""Process A (FD Table)      System Open File Table        Inode Table
[ FD 0 ]                  [ Entry X: Offset 0  ]  --->  [ Inode 123 ]
[ FD 3 ] ---------------->[ Entry Y: Offset 50 ]  --->  [ Inode 456 ]
                                     ^
Process B (FD Table)                 |
[ FD 3 ] ----------------------------+""")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '5. APIs & Directory Management', 0, 1)
pdf.chapter_body('Key APIs: open(), lseek(), fcntl() (locking).\nDirectory APIs: mkdir, rmdir, opendir, readdir.\nHard Link: Another name for the SAME inode.\nSymbolic Link: A file containing the PATH to another file.')

pdf.ln(5)

# Part 2
pdf.chapter_title('Part 2: Process Management')

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '1. Process Concept & PCB', 0, 1)
pdf.chapter_body('A Process is a program in execution. The kernel manages it using the Process Control Block (PCB).')
pdf.chapter_code("""+-------------------------+
| Process State (Run/Wait)|
| Process ID (PID)        |
| Program Counter (PC)    |
| CPU Registers           |
| Memory Management Info  |
| I/O Status Info         |
+-------------------------+""")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '2. Process States', 0, 1)
pdf.chapter_body('States: New -> Ready -> Running -> Waiting or Terminated.')
pdf.chapter_code("""   [New] -> [Ready] -> [Running] -> [Terminated]
               ^           |
               |           v
            [Waiting] <--- (I/O Event)""")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '3. Process Control (Fork Logic)', 0, 1)
pdf.chapter_body('fork() creates a duplicate process. It returns 0 to the child and the Child PID to the parent.')
pdf.chapter_code("""START
  |
fork()
  |
  +--- (PID == 0) --> Child Code (exec new program) -> exit()
  |
  +--- (PID > 0) ---> Parent Code -> wait() -> Continue""")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '4. Zombie vs Orphan', 0, 1)
pdf.chapter_body('Zombie: Child finished, parent has not called wait() yet.\nOrphan: Parent finished, child still running (adopted by init).')

# Output
pdf.output('OS_Notes.pdf')
print("PDF generated successfully: OS_Notes.pdf")