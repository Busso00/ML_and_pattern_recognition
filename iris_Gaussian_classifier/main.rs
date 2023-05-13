use std::fs::*;
use std::io::Read;
use std::path::*;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug)]
enum FileType {
    Text,
    Binary,
}

#[derive(Debug)]
struct File {
    name: String,
    content: Vec<u8>, // max 1000 bytes, rest of the file truncated
    creation_time: u64,
    type_: FileType,
}

#[derive(Debug)]
struct Dir{
    name: String,
    creation_time: u64,
    children: Vec<Node>,
}

#[derive(Debug)]
enum Node{
    File(File),
    Dir(Dir),
}

#[derive(Debug)]
struct FileSystem {
    root: Dir,
}

#[derive(Debug)]
struct MatchResult<'a>{
    queries: Vec<&'a str>,
    nodes: Vec<&'a mut Node>
}

fn current_time() -> Duration {
    let start = SystemTime::now();
    return start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
}

impl FileSystem {

    fn new() -> FileSystem {
        let root = FileSystem {
            root: Dir {
                name: String::from("root"),
                creation_time: current_time().as_secs(),
                children: Vec::new(),
            }
        };
        return root;
    }

    fn read_dir_r(path: &str, depth: usize, father_node: &mut Dir) {
		//foreach directory entry
        for entry in read_dir(path).unwrap() {
			//directory entry of OS (subnode of initial path)
            let dir = entry.unwrap();
			//path of directory entry
            let path = dir.path();
			//conversion to string of the path
            let path_str = path.to_str().unwrap();
			//take only filename
            let filename_str = path.file_name().unwrap().to_str().unwrap();
			//check if path is a directory
            if path.is_dir() {
                //create current node element
                let mut current_dir = Dir {
                    name: String::from(filename_str),
                    creation_time: current_time().as_secs(),
                    children: Vec::new(),
                };
                //recursively visit subdirectories to fill children vector
                FileSystem::read_dir_r(path_str, depth + 1, &mut current_dir);
                let current_node = Node::Dir(current_dir);
				//push into filesystem the current (already created) directory
                father_node.children.push(current_node);
            } else {
                //create current node element (need to read the file)
                let mut f = std::fs::File::open(path_str).unwrap();
				//create a buffer where to store file contents
                let mut buffer: Vec<u8> = Vec::new();
				//save the file contents
                f.read_to_end(&mut buffer).unwrap();
				//simple check for file type (actually more complex)
                let f_type = if &filename_str[filename_str.len() - 3..] == "txt" {
                    FileType::Text
                } else {
                    FileType::Binary
                };
				//create current node element
                let current_file = File {
                    name: String::from(filename_str),
                    content: buffer,
                    creation_time: current_time().as_secs(),
                    type_: f_type,
                };
                let current_node = Node::File(current_file);
				//push into file system current (already created) file
                father_node.children.push(current_node);
            }
        }
    }

    fn from_dir(path: &str) -> FileSystem {
		//create root of filesystem
        let mut root = FileSystem {
            root: Dir {
                name: String::from("root"),
                creation_time: current_time().as_secs(),
                children: Vec::new(),
            },
        };
		//recursively read directories
        FileSystem::read_dir_r(path, 0, &mut root.root);

        return root;
    }
    
    fn traverse_dirs(mut current_dir: &mut Dir, depth: usize, path_vec:Vec<String>)->Result<&mut Dir,&str>{
        //assume that path passed is the father directory 
		//of the element to CRUM
		//ex. to CRUM file A/B/C/filename
		//you can use this function to get &mut Dir to C
        for i in 0..depth {
            current_dir = 
                //inizio espressione di match
                match current_dir.children.iter_mut().find(|path_el| match path_el {Node::Dir(dir) => {if dir.name == path_vec[i] {true} else {false}},Node::File(_file) => false})
                {
                    Some(find)=>{
                        match find{
                            Node::Dir(dir)=> dir,
                            Node::File(_file)=>unreachable!()
                        }
                    },
                    None => {
                        return Err("Invalid path");
                    }
                };
        }
        return Ok(current_dir);
    }

    fn mk_dir(&mut self, path: &str) {
        //setup variables to traverse directories
        let path_buf = PathBuf::from(path);
        let length = path_buf.components().count();
        let path_vec: Vec<String> = path_buf
            .clone()
            .iter()
            .map(|path_el| path_el.to_string_lossy().to_string())
            .collect();

        //dummy value to check for path errors
        let dummy = &mut Dir{name:String::from(""),creation_time:0,children:Vec::new()};
        //depth of the new dir is len-1 because of countig first children dir in root as depth 0 
        //and to not try to traverse the new directory
        let current_dir = match FileSystem::traverse_dirs(&mut self.root, length-1, path_vec.clone()){
            Ok(dir)=>dir,
            Err(e) => {eprintln!("{}: {:?}",e,path); dummy}
        };
        //check for path errors
        if current_dir.name == ""{
            return ;
        }
        //create new directory
        let new_dir = Dir {
            name: path_vec[length - 1].clone(),
            creation_time: current_time().as_secs(),
            children: Vec::new(),
        };
        println!("Directory {:?} successfully created",path);
		//insert new directory
        current_dir.children.push(Node::Dir(new_dir));
    }

    fn rm_dir(&mut self,path: &str) {
        //setup variables for traverse directories
        let path_buf = PathBuf::from(path);
        let length = path_buf.components().count();
        let path_vec: Vec<String> = path_buf
            .clone()
            .iter()
            .map(|path_el| path_el.to_string_lossy().to_string())
            .collect();
        //dummy value to check for path errors
        let dummy = &mut Dir{name:String::from(""),creation_time:0,children:Vec::new()};
        //depth of the new dir is len-1 because of countig first children dir in root as depth 0 
        //and to not try to traverse the new directory
        let current_dir = match FileSystem::traverse_dirs(&mut self.root, length-1, path_vec.clone()){
            Ok(dir)=>dir,
            Err(e) => {eprintln!("{}: {:?}",e,path); dummy}
        };
        //check for path errors
        if current_dir.name == ""{
            return ;
        }
        
        let mut status = 1;
		//retain all children except the one (of type directory) whose name is the last in path (if empty)
        current_dir.children.retain(|node|{
            match node{
                Node::Dir(dir)=>{
                    if path_vec[length-1] == dir.name {
                        if dir.children.is_empty(){
                            status = 0;
                            false
                        }else{
							status = 2;
                            true
                        }
                    }else{
                        true
                    }
                },
                Node::File(_file)=>{true}
            }
        });
		//error (success) message
        if status == 0 {
            println!("Directory {:?} successfully removed",path);
        }else if status == 1{
            eprintln!("Directory {:?} not found",path);
        }else if status == 2{
			eprintln!("Directory {:?} isn't empty",path);
		}
    }

    fn new_file(&mut self, path: &str, file: File) {
        //setup variables for traverse directories
        let path_buf = PathBuf::from(path);
        let length = path_buf.components().count();
        let path_vec: Vec<String> = path_buf
            .clone()
            .iter()
            .map(|path_el| path_el.to_string_lossy().to_string())
            .collect();
        //dummy value to check for path errors
        let dummy = &mut Dir{name:String::from(""),creation_time:0,children:Vec::new()};
        //depth of the new dir is len because of countig first children dir in root as depth 0 
        //so to put in root path must be ""
        let current_dir = match FileSystem::traverse_dirs(&mut self.root, length, path_vec.clone()){
            Ok(dir)=>dir,
            Err(e) => {eprintln!("{}: {:?}",e,path); dummy}
        };
        //check for path errors
        if current_dir.name == ""{
            return ;
        };
        println!("File {:?} successfully created",file.name);
		//insert new file
        current_dir.children.push(Node::File(file));

    }
    
    fn rm_file(&mut self, path: &str) {
        //setup variables for traverse directories
        let path_buf = PathBuf::from(path);
        let length = path_buf.components().count();
        let path_vec: Vec<String> = path_buf
            .clone()
            .iter()
            .map(|path_el| path_el.to_string_lossy().to_string())
            .collect();
        //dummy value to check for path errors
        let dummy = &mut Dir{name:String::from(""),creation_time:0,children:Vec::new()};
        //depth of the new dir is len-1 because of countig first children dir in root as depth 0 
        //and to not try to traverse the new directory
        let current_dir = match FileSystem::traverse_dirs(&mut self.root, length-1, path_vec.clone()){
            Ok(dir)=>dir,
            Err(e) => {eprintln!("{}: {:?}",e,path); dummy}
        };
        //check for path errors
        if current_dir.name == ""{
            return ;
        };

		let mut status = 1;
        //retain all children except the one (of type file) whose name is the last in path
        current_dir.children.retain(|node|{
            match node{
                Node::Dir(_dir)=>{
                    true
                },
                Node::File(file)=>{
                    if path_vec[length-1] == file.name{
                        status = 0;
                        false
                    }else{
                        true
                    }
                }
            }
        });

        if status == 0 {
            println!("File {:?} successfully removed",path);
        }else if status == 1{
            eprintln!("File {:?} not found",path);
        }
    }

    fn get_file(&mut self,path: &str) -> Option<&mut File>{
        //setup variables for traverse directories
        let path_buf = PathBuf::from(path);
        let length = path_buf.components().count();
        let path_vec: Vec<String> = path_buf
            .clone()
            .iter()
            .map(|el| el.to_string_lossy().to_string())
            .collect();

        //no dummy value (cannot borrow out of function)

        //depth of the new dir is len-1 because of countig first children dir in root as depth 0 
        //and to not try to traverse the new directory
        match FileSystem::traverse_dirs(&mut self.root, length-1, path_vec.clone()){
            Ok(dir) => {
				//if path exist I still need to find the file in directory
                match dir.children.iter_mut().find(|path_el| match path_el {Node::Dir(_dir) => false,Node::File(file) => {if file.name == path_vec[length-1] {true} else {false}}})
                {
                    Some(find)=>{
                        match find{
                            Node::Dir(_dir)=> unreachable!(),
                            Node::File(file)=> Some(file)
                        }
                    },
                    None => {
                        eprintln!("File {:?} not found",path);
                        None
                    }
                }
            },
            Err(e) => {eprintln!("{}: {:?}",e,path); None}
        } 
    }  

    fn search<'a>(&'a mut self,queries: &[&'a str]) -> Option<MatchResult<'a>>{
        //struct to store query and results
        let mut mr=MatchResult{queries:Vec::new(),nodes:Vec::new()};
		//mut reference to all filesystem
        let mut unvisited_dir = vec![&mut self.root];
        //applying all the queries
        for query in queries{
			//parsing the query
            let query_s = query.split(":").collect::<Vec<&str>>();
			//all query stored in MatchResult
            mr.queries.push(query);
			//are there directory to visit?
            while unvisited_dir.len()>0{
				//current become first directory waiting to be visited --> pop
                let current = unvisited_dir.remove(0);
				//can obtain an itermut over children since current directory is mutable 
                for child in current.children.iter_mut(){
                    match child {
                        Node::File(file)=>{
                            //based on query field filter ([0]) I must perform different filtering on files
                            match query_s[0]{
								"name"=>{
    	                            if file.name == query_s[1] {
                                        mr.nodes.push(child)
                                    }
								},
								"content"=>{
									if String::from_utf8_lossy(&file.content[..]).contains(query_s[1]){
                                        mr.nodes.push(child)
                                	}
								},
								"larger"=>{
									let value=query_s[1].parse::<usize>().unwrap();
									if file.content.len() > value{
										mr.nodes.push(child)
									}
								},
								"smaller"=>{
									let value=query_s[1].parse::<usize>().unwrap();
									if file.content.len() < value{
										mr.nodes.push(child)
									}
								},
								"newer"=>{
									let value=query_s[1].parse::<u64>().unwrap();
									if file.creation_time < value{
										mr.nodes.push(child)
									}
								},
								"older"=>{
									let value=query_s[1].parse::<u64>().unwrap();
									if file.creation_time > value{
										mr.nodes.push(child)
									}
								},
								_=>{
									eprintln!("Invalid query"); 
									return None;
								}
                            }
                        },
                    	Node::Dir(dir) => {
							//BFS
							//in case matched node is a directory then I need to push it to the list of unvisited 
							//unvisited_dir now contains a reference to a child of the current directory
							//but this reference is not going to be visited (try to be owned by another variable than current) 
							//until all children are inserted into the unvisited list and so I can throw away current
							//--> only one mutable reference at is used at a time
							unvisited_dir.push(dir);
						}
                    }
                }
            }
        }
        if mr.nodes.len()>0 {
            //elimina duplicati per indirizzo
            //self.nodes=self.nodes.dedup();
            return Some(mr);
        }else{
            println!("No file found");
            return None;
        }
    }
}


fn main() {
    //test
    let root = FileSystem::new();
    println!("{:?}", root);
    let mut root = FileSystem::from_dir("prova");
    println!("initial root: {:?}", root);
    //test mk_dir()
    root.mk_dir("f1/nf11");
    root.mk_dir("f2/f21/f211/nf2111");
    root.mk_dir("nf3");
    root.mk_dir("nf4");
    root.mk_dir("f21/f211/f1/f2");
    root.mk_dir("f2/f21/f212/nf1/nf2");
    root.mk_dir("f2/f21/f211/nf5");
    root.mk_dir("f1/nf11/nf1n11");
    println!("root after mk_dir: {:?}", root);
    //test new_file()
    let file1 = File{
        name: String::from("prova1"),
        content: vec![122,121,123,110],
        creation_time: 2000,
        type_:FileType::Binary
    };
    let file2 = File{
        name: String::from("prova2"),
        content: vec![122,121,123,110],
        creation_time: 2000,
        type_:FileType::Binary
    };
    let file3 = File{
        name: String::from("prova3"),
        content: vec![122,121,123,110],
        creation_time: 2000,
        type_:FileType::Binary
    };
    let file4 = File{
        name: String::from("prova4"),
        content: vec![011,111,111,111],
        creation_time:1000,
        type_:FileType::Binary
    };
    let file5 = File{
        name: String::from("prova5"),
        content: vec![011,111,111,111],
        creation_time:1000,
        type_:FileType::Binary
    };
    let file6 = File{
        name: String::from("prova6"),
        content: vec![011,111,111,111],
        creation_time:1000,
        type_:FileType::Binary
    };
    let file7 = File{
        name: String::from("fileroot"),
        content: vec![011,111,111,111],
        creation_time:1000,
        type_:FileType::Binary
    };
    let file8 = File{
        name: String::from("filetxt.txt"),
        content: vec![0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49],
        creation_time:1000,
        type_:FileType::Text
    };
    //test new_file
    root.new_file("f1/nf11",file1);
    root.new_file("nf3",file2);
    root.new_file("",file3);
    root.new_file("nf3/prova",file4);
    root.new_file("f1/nf11",file5);
    root.new_file("nf3",file6);
    root.new_file("",file7);
    root.new_file("f2/f21/f211/nf5",file8);
    println!("root after new_file: {:?}", root);
    //test rm_dir
    root.rm_dir("nf3");
    root.rm_dir("f2");
    root.rm_dir("f1/nf11/nf1n11");
    root.rm_dir("nf4");
    root.rm_dir("f2/f21/f211/nf5");
    root.rm_dir("f2/f21/prova/nf5");
    root.rm_dir("f2/f21/prova");
    println!("root after rm_dir: {:?}", root);
    //test rm_file
    root.rm_file("f1/nf11/prova1");
    root.rm_file("prova3");
    root.rm_file("nf3/prova2");
    root.rm_file("nf3/prova2/file4");
    root.rm_file("f1/nf11");
    println!("root after rm_file: {:?}", root);
    //test get_file
    let file=root.get_file("nf3/prova/prova4");
    println!("file: {:?}", file);
    let file=root.get_file("f1/nf11/prova5");
    println!("file: {:?}", file);
    let file=root.get_file("nf3/prova6");
    println!("file: {:?}", file);
    let file=root.get_file("f1/nf11/no/nope");
    println!("file: {:?}", file);
    let file=root.get_file("nf4/fileno");
    println!("file: {:?}", file);
    let file=root.get_file("fileroot");
    println!("file: {:?}", file);
    //test search
    println!("test query multiple larger 5, newer 999");
    let args=["larger:5","newer:1001"];
    root.search(&args);
    let args=["name:prova5"];
    let res=root.search(&args);
    println!("{:?}",res);
    let args=["content:ABCDEFGHI"];
    let res=root.search(&args);
    println!("{:?}",res);
}
