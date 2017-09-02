import time
import MainFunction
import Config


def main():
    
    input_path = ''
    output_path = ''
    
    MainFunction.PatternMining( input_path, output_path, Config.parameters() )
    
    print 'Hello World!'


if __name__ == '__main__' :
    
    t_start_time = time.time()
    
    main()
   
    print 'All finished in ',time.time()-t_start_time,' s'