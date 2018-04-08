package com.tooth.tooth;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Paths;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class Principal {
	
	@Autowired
	LabelImage lableImage;
	
	 @RequestMapping("/")
	  public String index() {
	    return "index.html";
	  }
	
	@RequestMapping("hello")
	public String sayHello(){
		return "Hello";
	}
	
	  @RequestMapping(value = "/uploadFile", method = RequestMethod.POST)
	  @ResponseBody
	  public String uploadFile(
	      @RequestParam("uploadfile") MultipartFile uploadfile) {
		  String result = "";
	    try {
	      // Get the filename and build the local file path
	      String filename = uploadfile.getOriginalFilename();
	      String directory = "C:\\Users\\User\\Desktop\\faces";//env.getProperty("netgloo.paths.uploadedFiles");
	      String filepath = Paths.get(directory, filename).toString();
	      result = lableImage.process(uploadfile.getBytes());	     
	      System.out.println("RESULTADO "+result);
	      // Save the file locally
	      //BufferedOutputStream stream =
	      //  new BufferedOutputStream(new FileOutputStream(new File(filepath)));
	      //stream.write(uploadfile.getBytes());
	      //stream.close();
	    }
	    catch (Exception e) {
	      System.out.println(e.getMessage());
	      //return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
	    }
	    
	    return result;
	  } // method uploadFile

}
