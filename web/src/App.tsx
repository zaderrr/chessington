import './App.css'
import MoveBoard from './assets/components/moveBoard'
import Board from './assets/components/board'
function App() {

    return (
        <div className='flex flex-col w-full h-full'>
            <div className="w-full flex flex-row justify-center">
                <h1 className="text-3xl">Play with chessington!</h1>
            </div>
            <div className="w-full flex h-full flex-col justify-center">
                <div className='w-full  flex flex-row h-full justify-center'>
                    <div className="flex flex-row w-1/5 flex-3  justify-center ">
                        <Board />
                    </div>
                    <div className="flex h-full flex-row w-1/4 ">
                        <MoveBoard />
                    </div>
                </div>
            </div>
        </div >
    )
}

export default App
