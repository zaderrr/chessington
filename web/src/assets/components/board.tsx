import { Chessboard, type ChessboardOptions } from 'react-chessboard';
import ErrorBoundary from './ErrorBoundary';


const Board = () => {

    const boardSetup: ChessboardOptions = {
        boardStyle: { width: '50%' }
    }

    return (

        <ErrorBoundary>
            <div className='flex flex-col justify-center w-full '>
                <div className='flex flex-row justify-center'>
                    <Chessboard options={boardSetup} />
                </div>
            </div>
        </ErrorBoundary>
    );
}

export default Board;
